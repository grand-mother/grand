//! This Zig build script compile the two C library required by GRANDLib
//! (gull and turtle) into static libraries that are then used in GRANDLib
//! C code and exported to python via cffi.
const std = @import("std");
const Version = std.SemanticVersion;
const builtin = @import("builtin");

const zig16: []const u8 = "0.16.0";
const zig17: []const u8 = "0.17.0";

const zig_version = builtin.zig_version;
const zig_version_range: Version.Range = .{
    .min = Version.parse(zig16) catch unreachable,
    .max = Version.parse(zig17) catch unreachable,
};

// Comptime check that the Zig compiler version being used to compile this
// script meets the targeted version. If not, emits a compile error.
// This force the required Zig version to be 0.16.X as Zig is still in an
// "early" development stage and backward compatibility is not expected
// before version 1.0.
comptime {
    if (!zig_version_range.includesVersion(zig_version)) {
        const error_msg = std.fmt.comptimePrint("Requires Zig version between {s} and {s}, got {d}.{d}.{d}\n", .{
            zig16,
            zig17,
            zig_version.major,
            zig_version.minor,
            zig_version.patch,
        });
        @compileError(error_msg);
    }
}

// Entry point of the build script.
pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Add turtle to the build graph to be compiled as a static library.
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

    // Add gull to the build graph to be compiled as a static library.
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

    // Install libraries and headers files
    b.installArtifact(turtle);
    b.installArtifact(gull);
    b.installFile("build/turtle/include/turtle.h", "include/turtle.h");
    b.installFile("build/gull/include/gull.h", "include/gull.h");
}
