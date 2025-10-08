find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(NUMERIC_BASIS_LAGRANGE_MAX_ORDER 5 CACHE STRING "Maximum order of lagrangian basis functions")

set(NUMERIC_ELEMENT_NAMES "segment;tria;quad;tetra;cube")

set(BASIS_LAGRANGE_SOURCE_FILE ${NUMERIC_INCLUDE_DIR}/numeric/math/basis_lagrange_specialization.hpp)
execute_process(
  COMMAND ${Python3_EXECUTABLE} basis_lagrange.py ${NUMERIC_BASIS_LAGRANGE_MAX_ORDER}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts
  OUTPUT_VARIABLE BASIS_LAGRANGE_SRC
  COMMAND_ECHO STDOUT
)
file(
  GENERATE OUTPUT ${BASIS_LAGRANGE_SOURCE_FILE}
  CONTENT "${BASIS_LAGRANGE_SRC}"
)
add_custom_command(
  OUTPUT ${BASIS_LAGRANGE_SOURCE_FILE}
  COMMAND ${Python3_EXECUTABLE} basis_lagrange.py ${NUMERIC_BASIS_LAGRANGE_MAX_ORDER} > ${BASIS_LAGRANGE_SOURCE_FILE}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts
  DEPENDS ${CMAKE_SOURCE_DIR}/scripts/basis_lagrange.py
	  ${CMAKE_SOURCE_DIR}/scripts/elements.py
	  COMMENT Generating ${BASIS_LAGRANGE_SOURCE_FILE}
)
add_custom_target(generate_basis_lagrange DEPENDS ${BASIS_LAGRANGE_SOURCE_FILE})
add_dependencies(numeric generate_basis_lagrange)
