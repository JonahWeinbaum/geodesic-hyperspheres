cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(GeodesicHyperspheres
	LANGUAGES CXX
)

set(CPM_DOWNLOAD_VERSION 0.27.2) 
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")

if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(STATUS "Downloading CPM.cmake")
    file(DOWNLOAD https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake ${CPM_DOWNLOAD_LOCATION})
endif()

include(${CPM_DOWNLOAD_LOCATION})

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Define sources and headers
set(SOURCES src/main.cpp)

# Add the linalg library using CPMAddPackage
CPMAddPackage(
  NAME linalg
  GITHUB_REPOSITORY sgorsten/linalg
  GIT_TAG a3e87da35e32b781a4b6c01cdd5efbe7ae51c737
 )

# If linalg was added successfully, configure it
if(linalg_ADDED)
  add_library(linalg INTERFACE IMPORTED)
  target_include_directories(linalg INTERFACE "${linalg_SOURCE_DIR}")
endif()


# Add executable
add_executable(GeodesicHyperspheres ${SOURCES})

target_include_directories(GeodesicHyperspheres PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link the linalg library to the executable
target_link_libraries(GeodesicHyperspheres PRIVATE linalg)

set_target_properties(GeodesicHyperspheres PROPERTIES CXX_STANDARD 11)
