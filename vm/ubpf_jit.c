// Copyright (c) 2015 Big Switch Networks, Inc
// SPDX-License-Identifier: Apache-2.0

/*
 * Copyright 2015 Big Switch Networks, Inc
 * Copyright 2017 Google Inc.
 * Copyright 2022 Linaro Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ubpf.h"
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>
#if defined(__CHERI_PURE_CAPABILITY__)
#include <cheriintrin.h>
#include <dlfcn.h>
#endif
#include "ubpf_int.h"

int
ubpf_translate_ex(struct ubpf_vm* vm, uint8_t* buffer, size_t* size, char** errmsg, enum JitMode jit_mode)
{
    struct ubpf_jit_result jit_result = vm->jit_translate(vm, buffer, size, jit_mode);
    vm->jitted_result = jit_result;
    if (jit_result.errmsg) {
        *errmsg = jit_result.errmsg;
    }
    return jit_result.compile_result == UBPF_JIT_COMPILE_SUCCESS ? 0 : -1;
}

int
ubpf_translate(struct ubpf_vm* vm, uint8_t* buffer, size_t* size, char** errmsg)
{
    return ubpf_translate_ex(vm, buffer, size, errmsg, BasicJitMode);
}

struct ubpf_jit_result
ubpf_translate_null(struct ubpf_vm* vm, uint8_t* buffer, size_t* size, enum JitMode jit_mode)
{
    struct ubpf_jit_result compile_result;
    compile_result.compile_result = UBPF_JIT_COMPILE_FAILURE;
    compile_result.external_dispatcher_offset = 0;

    /* NULL JIT target - just returns an error. */
    UNUSED_PARAMETER(vm);
    UNUSED_PARAMETER(buffer);
    UNUSED_PARAMETER(size);
    UNUSED_PARAMETER(jit_mode);
    compile_result.errmsg = ubpf_error("Code can not be JITed on this target.");
    return compile_result;
}

bool
ubpf_jit_update_dispatcher_null(
    struct ubpf_vm* vm, external_function_dispatcher_t new_dispatcher, uint8_t* buffer, size_t size, uint32_t offset)
{
    UNUSED_PARAMETER(vm);
    UNUSED_PARAMETER(new_dispatcher);
    UNUSED_PARAMETER(buffer);
    UNUSED_PARAMETER(size);
    UNUSED_PARAMETER(offset);
    return false;
}

bool
ubpf_jit_update_helper_null(
    struct ubpf_vm* vm,
    extended_external_helper_t new_helper,
    unsigned int idx,
    uint8_t* buffer,
    size_t size,
    uint32_t offset)
{
    UNUSED_PARAMETER(vm);
    UNUSED_PARAMETER(new_helper);
    UNUSED_PARAMETER(idx);
    UNUSED_PARAMETER(buffer);
    UNUSED_PARAMETER(size);
    UNUSED_PARAMETER(offset);
    return false;
}

#if defined(__CHERI_PURE_CAPABILITY__)
static void
ubpf_close_cheri_objjit(struct ubpf_vm* vm)
{
    if (vm->cheri_objjit_handle) {
        dlclose(vm->cheri_objjit_handle);
        vm->cheri_objjit_handle = NULL;
    }
}

static bool
ubpf_cheri_objjit_supported_context_load(const struct ubpf_vm* vm, int16_t* offset_out, char** errmsg)
{
    uint8_t context_reg = 1;
    uint32_t load_pc = 0;
    uint32_t exit_pc = 1;

    if (vm->num_insts == 3) {
        struct ebpf_inst alias = ubpf_fetch_instruction(vm, 0);
        if (alias.opcode != EBPF_OP_MOV64_REG || alias.src != 1 || alias.dst < 2 || alias.dst > 9 ||
            alias.offset != 0 || alias.imm != 0) {
            *errmsg = ubpf_error(
                "CHERI object JIT only supports a leading rN = r1 alias before context loads");
            return false;
        }
        context_reg = alias.dst;
        load_pc = 1;
        exit_pc = 2;
    } else if (vm->num_insts != 2) {
        *errmsg = ubpf_error(
            "CHERI object JIT only supports direct or single-alias context-load programs");
        return false;
    }

    struct ebpf_inst load = ubpf_fetch_instruction(vm, load_pc);
    struct ebpf_inst exit_inst = ubpf_fetch_instruction(vm, exit_pc);

    if (load.opcode != EBPF_OP_LDXDW || load.dst != 0 || load.src != context_reg || load.imm != 0) {
        *errmsg = ubpf_error(
            "CHERI object JIT only supports r0 = *(u64 *)(context + offset); got opcode %02x dst %u src %u imm %d",
            load.opcode,
            load.dst,
            load.src,
            load.imm);
        return false;
    }

    if (exit_inst.opcode != EBPF_OP_EXIT || exit_inst.dst != 0 || exit_inst.src != 0 || exit_inst.offset != 0 ||
        exit_inst.imm != 0) {
        *errmsg = ubpf_error("CHERI object JIT requires a plain EXIT after the context load");
        return false;
    }

    if (load.offset < 0) {
        *errmsg = ubpf_error("CHERI object JIT does not yet support negative context offsets");
        return false;
    }
    if ((load.offset % 8) != 0) {
        *errmsg = ubpf_error("CHERI object JIT context-load offset must be 8-byte aligned");
        return false;
    }
    if ((load.offset / 8) >= 0x1000) {
        *errmsg = ubpf_error("CHERI object JIT context-load offset is too large for the current Morello encoding");
        return false;
    }

    *offset_out = load.offset;
    return true;
}

static ubpf_jit_ex_fn
ubpf_compile_cheri_objjit(struct ubpf_vm* vm, char** errmsg, enum JitMode mode)
{
    if (mode != BasicJitMode) {
        return NULL;
    }

    int16_t offset = 0;
    char* shape_error = NULL;
    if (!ubpf_cheri_objjit_supported_context_load(vm, &offset, &shape_error)) {
        free(shape_error);
        return NULL;
    }

    const char* dir = getenv("UBPF_CHERI_OBJJIT_DIR");
    if (!dir || dir[0] == '\0') {
        dir = "/mnt";
    }

    char path[256];
    int written = snprintf(path, sizeof(path), "%s/cheri_objjit_offset_%d.so", dir, (int)offset);
    if (written < 0 || (size_t)written >= sizeof(path)) {
        *errmsg = ubpf_error("CHERI object JIT artifact path is too long");
        return NULL;
    }

    void* handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        *errmsg = ubpf_error("CHERI object JIT failed to dlopen %s: %s", path, dlerror());
        return NULL;
    }

    dlerror();
    void* entry = dlsym(handle, "bpf_entry");
    const char* dlsym_error = dlerror();
    if (dlsym_error) {
        *errmsg = ubpf_error("CHERI object JIT failed to resolve bpf_entry in %s: %s", path, dlsym_error);
        dlclose(handle);
        return NULL;
    }

    ubpf_close_cheri_objjit(vm);
    vm->cheri_objjit_handle = handle;
    vm->jitted_mapping = NULL;
    vm->jitted = (ubpf_jit_ex_fn)entry;
    vm->jitted_size = 0;
    vm->jitted_result.compile_result = UBPF_JIT_COMPILE_SUCCESS;
    vm->jitted_result.jit_mode = mode;
    vm->jitted_result.errmsg = NULL;
    vm->jitted_result.external_dispatcher_offset = 0;
    vm->jitted_result.external_helper_offset = 0;
    *errmsg = NULL;
    return vm->jitted;
}
#endif

int
ubpf_set_jit_code_size(struct ubpf_vm* vm, size_t code_size)
{
    vm->jitter_buffer_size = code_size;
    return 0;
}

ubpf_jit_fn
ubpf_compile(struct ubpf_vm* vm, char** errmsg)
{
    return (ubpf_jit_fn)ubpf_compile_ex(vm, errmsg, BasicJitMode);
}

ubpf_jit_ex_fn
ubpf_compile_ex(struct ubpf_vm* vm, char** errmsg, enum JitMode mode)
{
    void* jitted = NULL;
    uint8_t* buffer = NULL;
    size_t jitted_size;

    if (vm->jitted && vm->jitted_result.compile_result == UBPF_JIT_COMPILE_SUCCESS &&
        vm->jitted_result.jit_mode == mode) {
        return vm->jitted;
    }

#if defined(__CHERI_PURE_CAPABILITY__)
    if (vm->cheri_objjit_handle) {
        ubpf_close_cheri_objjit(vm);
        vm->jitted = NULL;
        vm->jitted_mapping = NULL;
        vm->jitted_size = 0;
    } else
#endif
    if (vm->jitted) {
#if defined(__CHERI_PURE_CAPABILITY__)
        munmap(vm->jitted_mapping, vm->jitted_size);
        vm->jitted_mapping = NULL;
#else
        munmap(vm->jitted, vm->jitted_size);
#endif
        vm->jitted = NULL;
        vm->jitted_size = 0;
    }

    *errmsg = NULL;

    if (!vm->insts) {
        *errmsg = ubpf_error("code has not been loaded into this VM");
        return NULL;
    }

#if defined(__CHERI_PURE_CAPABILITY__)
    const char* use_objjit = getenv("UBPF_CHERI_USE_OBJJIT");
    if (use_objjit && strcmp(use_objjit, "1") == 0) {
        ubpf_jit_ex_fn objjit = ubpf_compile_cheri_objjit(vm, errmsg, mode);
        if (objjit) {
            return objjit;
        }
        if (*errmsg) {
            return NULL;
        }
    }
#endif

    jitted_size = vm->jitter_buffer_size;
    buffer = calloc(jitted_size, 1);
    if (buffer == NULL) {
        *errmsg = ubpf_error("internal uBPF error: calloc failed: %s\n", strerror(errno));
        goto out;
    }

    if (ubpf_translate_ex(vm, buffer, &jitted_size, errmsg, mode) < 0) {
        goto out;
    }

#ifdef __CHERI_PURE_CAPABILITY__
    /* CHERI M3: allocate a separate PROT_READ|PROT_WRITE page for the
     * eBPF stack. This avoids the W^X SIGPROT crash that occurs when
     * stores are executed from mmap'd PROT_EXEC memory on QEMU Morello.
     * Patch the literal pool entry in the JIT buffer with the stack top
     * address (base + size), since eBPF stack grows downward from R10. */
    void* ebpf_stack_page = NULL;
    if (mode == BasicJitMode && vm->jitted_result.stack_base_offset > 0) {
        ebpf_stack_page = mmap(0, UBPF_EBPF_STACK_SIZE,
                               PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ebpf_stack_page == MAP_FAILED) {
            *errmsg = ubpf_error("internal uBPF error: stack mmap failed: %s\n", strerror(errno));
            goto out;
        }
        vm->ebpf_stack_page = ebpf_stack_page;
        uint64_t stack_top = (uint64_t)ebpf_stack_page + UBPF_EBPF_STACK_SIZE;
        uint32_t off = vm->jitted_result.stack_base_offset;
        memcpy(buffer + off, &stack_top, sizeof(uint64_t));
    }
#endif

    int mmap_prot = PROT_READ | PROT_WRITE;
    int mprotect_prot = PROT_READ | PROT_EXEC;
#ifdef __CHERI_PURE_CAPABILITY__
    /* CheriBSD purecap permissions are carried by capabilities. Reserve the
     * complete RWX+CAP maximum at allocation time, then narrow the mapping to
     * RX+CAP after copying the translated code. */
    mmap_prot = PROT_READ | PROT_WRITE | PROT_EXEC | PROT_CAP;
    mprotect_prot = PROT_READ | PROT_EXEC | PROT_CAP;
    mmap_prot |= PROT_MAX(PROT_READ | PROT_WRITE | PROT_EXEC | PROT_CAP);
#endif
    jitted = mmap(0, jitted_size, mmap_prot, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (jitted == MAP_FAILED) {
        *errmsg = ubpf_error("internal uBPF error: mmap failed: %s\n", strerror(errno));
        goto out;
    }

    memcpy(jitted, buffer, jitted_size);

    if (mprotect(jitted, jitted_size, mprotect_prot) < 0) {
        *errmsg = ubpf_error("internal uBPF error: mprotect failed: %s\n", strerror(errno));
        goto out;
    }

#ifdef __CHERI_PURE_CAPABILITY__
    vm->jitted_mapping = jitted;
    vm->jitted = (ubpf_jit_ex_fn)cheri_sentry_create((void*)((uintptr_t)jitted | 1U));
#else
    vm->jitted = jitted;
#endif
    vm->jitted_size = jitted_size;

out:
    free(buffer);
    if (jitted && vm->jitted == NULL) {
        munmap(jitted, jitted_size);
    }
    return vm->jitted;
}

ubpf_jit_fn
ubpf_copy_jit(struct ubpf_vm* vm, void* buffer, size_t size, char** errmsg)
{
    // If compilation was not successfull or it has not even been attempted,
    // we cannot copy.
    if (vm->jitted_result.compile_result != UBPF_JIT_COMPILE_SUCCESS || !vm->jitted) {
        *errmsg = ubpf_error("Cannot copy JIT'd code before compilation");
        return (ubpf_jit_fn)NULL;
    }

    // If the given buffer is not big enough to contain the JIT'd code,
    // we cannot copy.
    if (vm->jitted_size > size) {
        *errmsg = ubpf_error("Buffer not big enough for copy");
        return (ubpf_jit_fn)NULL;
    }

    // All good. Do the copy!
#if defined(__CHERI_PURE_CAPABILITY__)
    memcpy(buffer, vm->jitted_mapping, vm->jitted_size);
#else
    memcpy(buffer, vm->jitted, vm->jitted_size);
#endif
    *errmsg = NULL;
    return (ubpf_jit_fn)buffer;
}
