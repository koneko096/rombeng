cmake_minimum_required(VERSION 3.21)
project(rombeng_kernel)

add_custom_command(
  OUTPUT ${CMAKE_SOURCE_DIR}/dotproduct.air ${CMAKE_SOURCE_DIR}/matmul.air
  DEPENDS ${CMAKE_SOURCE_DIR}/dotproduct.metal ${CMAKE_SOURCE_DIR}/matmul.metal
  COMMENT "Building IR"
  COMMAND xcrun -sdk macosx metal -c ${CMAKE_SOURCE_DIR}/dotproduct.metal -o ${CMAKE_SOURCE_DIR}/dotproduct.air
  COMMAND xcrun -sdk macosx metal -c ${CMAKE_SOURCE_DIR}/matmul.metal -o ${CMAKE_SOURCE_DIR}/matmul.air
  VERBATIM)

add_custom_command(
	OUTPUT ${CMAKE_SOURCE_DIR}/dotproduct.metallib ${CMAKE_SOURCE_DIR}/matmul.metallib
	DEPENDS ${CMAKE_SOURCE_DIR}/dotproduct.air ${CMAKE_SOURCE_DIR}/matmul.air
	COMMENT "Generate metal lib"
	COMMAND xcrun -sdk macosx metallib ${CMAKE_SOURCE_DIR}/dotproduct.air -o ${CMAKE_SOURCE_DIR}/dotproduct.metallib
	COMMAND xcrun -sdk macosx metallib ${CMAKE_SOURCE_DIR}/matmul.air -o ${CMAKE_SOURCE_DIR}/matmul.metallib
	VERBATIM)

add_custom_target(metal ALL
  DEPENDS ${CMAKE_SOURCE_DIR}/dotproduct.metallib ${CMAKE_SOURCE_DIR}/matmul.metallib)

install(FILES ${CMAKE_SOURCE_DIR}/dotproduct.metallib ${CMAKE_SOURCE_DIR}/matmul.metallib DESTINATION .)