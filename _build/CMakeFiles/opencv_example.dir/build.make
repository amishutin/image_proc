# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /snap/cmake/1366/bin/cmake

# The command to remove a file.
RM = /snap/cmake/1366/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lev/image_proc

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lev/image_proc/_build

# Include any dependencies generated for this target.
include CMakeFiles/opencv_example.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/opencv_example.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/opencv_example.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/opencv_example.dir/flags.make

CMakeFiles/opencv_example.dir/example.cpp.o: CMakeFiles/opencv_example.dir/flags.make
CMakeFiles/opencv_example.dir/example.cpp.o: /home/lev/image_proc/example.cpp
CMakeFiles/opencv_example.dir/example.cpp.o: CMakeFiles/opencv_example.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/lev/image_proc/_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/opencv_example.dir/example.cpp.o"
	/opt/riscv/bin/riscv64-unknown-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/opencv_example.dir/example.cpp.o -MF CMakeFiles/opencv_example.dir/example.cpp.o.d -o CMakeFiles/opencv_example.dir/example.cpp.o -c /home/lev/image_proc/example.cpp

CMakeFiles/opencv_example.dir/example.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/opencv_example.dir/example.cpp.i"
	/opt/riscv/bin/riscv64-unknown-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lev/image_proc/example.cpp > CMakeFiles/opencv_example.dir/example.cpp.i

CMakeFiles/opencv_example.dir/example.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/opencv_example.dir/example.cpp.s"
	/opt/riscv/bin/riscv64-unknown-linux-gnu-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lev/image_proc/example.cpp -o CMakeFiles/opencv_example.dir/example.cpp.s

# Object files for target opencv_example
opencv_example_OBJECTS = \
"CMakeFiles/opencv_example.dir/example.cpp.o"

# External object files for target opencv_example
opencv_example_EXTERNAL_OBJECTS =

opencv_example: CMakeFiles/opencv_example.dir/example.cpp.o
opencv_example: CMakeFiles/opencv_example.dir/build.make
opencv_example: /home/lev/_opencv_c910v/lib/libopencv_calib3d.a
opencv_example: /home/lev/_opencv_c910v/lib/libopencv_core.a
opencv_example: /home/lev/_opencv_c910v/lib/libopencv_dnn.a
opencv_example: /home/lev/_opencv_c910v/lib/libopencv_features2d.a
opencv_example: /home/lev/_opencv_c910v/lib/libopencv_flann.a
opencv_example: /home/lev/_opencv_c910v/lib/libopencv_gapi.a
opencv_example: /home/lev/_opencv_c910v/lib/libopencv_highgui.a
opencv_example: /home/lev/_opencv_c910v/lib/libopencv_imgcodecs.a
opencv_example: /home/lev/_opencv_c910v/lib/libopencv_imgproc.a
opencv_example: /home/lev/_opencv_c910v/lib/libopencv_ml.a
opencv_example: /home/lev/_opencv_c910v/lib/libopencv_objdetect.a
opencv_example: /home/lev/_opencv_c910v/lib/libopencv_photo.a
opencv_example: /home/lev/_opencv_c910v/lib/libopencv_stitching.a
opencv_example: /home/lev/_opencv_c910v/lib/libopencv_video.a
opencv_example: /home/lev/_opencv_c910v/lib/libopencv_videoio.a
opencv_example: /home/lev/_opencv_c910v/3rdparty/lib/libade.a
opencv_example: /home/lev/_opencv_c910v/lib/libopencv_imgcodecs.a
opencv_example: /home/lev/_opencv_c910v/3rdparty/lib/liblibjpeg-turbo.a
opencv_example: /home/lev/_opencv_c910v/3rdparty/lib/liblibwebp.a
opencv_example: /home/lev/_opencv_c910v/3rdparty/lib/liblibpng.a
opencv_example: /home/lev/_opencv_c910v/3rdparty/lib/liblibtiff.a
opencv_example: /home/lev/_opencv_c910v/3rdparty/lib/liblibopenjp2.a
opencv_example: /home/lev/_opencv_c910v/lib/libopencv_dnn.a
opencv_example: /home/lev/_opencv_c910v/3rdparty/lib/liblibprotobuf.a
opencv_example: /home/lev/_opencv_c910v/lib/libopencv_calib3d.a
opencv_example: /home/lev/_opencv_c910v/lib/libopencv_features2d.a
opencv_example: /home/lev/_opencv_c910v/lib/libopencv_flann.a
opencv_example: /home/lev/_opencv_c910v/lib/libopencv_imgproc.a
opencv_example: /home/lev/_opencv_c910v/lib/libopencv_core.a
opencv_example: /home/lev/_opencv_c910v/3rdparty/lib/libzlib.a
opencv_example: CMakeFiles/opencv_example.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/lev/image_proc/_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable opencv_example"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/opencv_example.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/opencv_example.dir/build: opencv_example
.PHONY : CMakeFiles/opencv_example.dir/build

CMakeFiles/opencv_example.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/opencv_example.dir/cmake_clean.cmake
.PHONY : CMakeFiles/opencv_example.dir/clean

CMakeFiles/opencv_example.dir/depend:
	cd /home/lev/image_proc/_build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lev/image_proc /home/lev/image_proc /home/lev/image_proc/_build /home/lev/image_proc/_build /home/lev/image_proc/_build/CMakeFiles/opencv_example.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/opencv_example.dir/depend
