const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // libturtle
    const turtle = b.addLibrary(.{
        .name = "turtle",
        .linkage = .static,
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    turtle.root_module.addCSourceFiles(.{
        .files = &.{
            "build/turtle/src/deps/jsmn.c",
            "build/turtle/src/deps/tinydir.c",
            "build/turtle/src/turtle/client.c",
            "build/turtle/src/turtle/ecef.c",
            "build/turtle/src/turtle/error.c",
            "build/turtle/src/turtle/io.c",
            "build/turtle/src/turtle/io/asc.c",
            "build/turtle/src/turtle/io/grd.c",
            "build/turtle/src/turtle/io/hgt.c",
            "build/turtle/src/turtle/io/png16.c",
            "build/turtle/src/turtle/list.c",
            "build/turtle/src/turtle/map.c",
            "build/turtle/src/turtle/projection.c",
            "build/turtle/src/turtle/stack.c",
            "build/turtle/src/turtle/stepper.c",
        },
        .flags = &.{
            "-DTURTLE_NO_TIFF",
            "-std=c99",
            "-pedantic",
        },
    });
    turtle.root_module.addIncludePath(b.path("build/turtle/include"));
    turtle.root_module.addIncludePath(b.path("build/turtle/src"));
    turtle.root_module.addIncludePath(b.path("build/turtle/src/deps"));

    // libgull
    const gull = b.addLibrary(.{
        .name = "gull",
        .linkage = .static,
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    gull.root_module.addCSourceFiles(.{
        .files = &.{
            "build/gull/src/gull.c",
        },
        .flags = &.{
            "-std=c99",
            "-pedantic",
        },
    });
    gull.root_module.addIncludePath(b.path("build/gull/include"));
    gull.root_module.addIncludePath(b.path("build/turtle/include"));

    // install libs and headers
    b.installArtifact(turtle);
    b.installArtifact(gull);
    b.installFile("build/turtle/include/turtle.h", "include/turtle.h");
    b.installFile("build/gull/include/gull.h", "include/gull.h");

    //    // grand/_core.abi3.so via zig cc
    //
    //    _ = b.run(&.{ "mkdir", "-p", "build/grand" });
    //
    //    //    const python_include = "/Users/mregeard/anaconda3/envs/grand/include/python3.9";
    //    const python_include = b.run(&.{ "python3", "-c", "import sysconfig; print(sysconfig.get_path('include'), end='')" });
    //
    //    const compile_core = b.addSystemCommand(&.{
    //        "zig",      "cc",
    //        "-target",  "x86_64-macos",
    //        "-std=c99", "-include",
    //        "stdlib.h", "-include",
    //        "string.h", "-include",
    //        "math.h",   "-O3",
    //        "-shared",  "-fPIC",
    //        "-undefined",                     "dynamic_lookup", // ← macOS: let Python symbols resolve at runtime
    //        "-Wl,-rpath,@loader_path/../lib", "-I",
    //        "build/turtle/include",           "-I",
    //        "build/gull/include",             "-I",
    //        python_include,                   "grand.c",
    //        "-L",                             "build/lib",
    //        "-lturtle",                       "-lgull",
    //        "-o", "build/grand/_core.abi3.so", // ← exact name, no lib prefix
    //    });
    //    compile_core.step.dependOn(&b.addInstallArtifact(turtle, .{}).step);
    //    compile_core.step.dependOn(&b.addInstallArtifact(gull, .{}).step);
    //    b.getInstallStep().dependOn(&compile_core.step);
}
