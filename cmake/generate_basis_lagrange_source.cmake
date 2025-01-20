find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(NUMERIC_BASIS_LAGRANGE_MAX_ORDER 5 CACHE STRING "Maximum order of lagrangian basis functions")

set(NUMERIC_ELEMENT_NAMES "segment;tria;quad;tetra;cube")

set(NUMERIC_GENERATED_FILES)
foreach (ELEMENT ${NUMERIC_ELEMENT_NAMES})
  set(ELEMENT_BASIS_SOURCE_FILE ${NUMERIC_INCLUDE_DIR}/numeric/math/basis_lagrange_${ELEMENT}.hpp)
  list(APPEND NUMERIC_GENERATED_FILES ${ELEMENT_BASIS_SOURCE_FILE})
  execute_process(
    COMMAND ${Python3_EXECUTABLE} basis_lagrange.py ${ELEMENT} ${NUMERIC_BASIS_LAGRANGE_MAX_ORDER}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts
    OUTPUT_VARIABLE ${ELEMENT}_SRC
    COMMAND_ECHO STDOUT
  )
  file(
    GENERATE OUTPUT ${ELEMENT_BASIS_SOURCE_FILE}
    CONTENT "${${ELEMENT}_SRC}"
  )
  add_custom_command(
    OUTPUT ${ELEMENT_BASIS_SOURCE_FILE}
    COMMAND ${Python3_EXECUTABLE} basis_lagrange.py ${ELEMENT} ${NUMERIC_BASIS_LAGRANGE_MAX_ORDER} > ${ELEMENT_BASIS_SOURCE_FILE}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts
    DEPENDS ${CMAKE_SOURCE_DIR}/scripts/basis_lagrange.py
    COMMENT Generating ${ELEMENT_BASIS_SOURCE_FILE}
  )
  add_custom_target(generate_${ELEMENT} DEPENDS ${ELEMENT_BASIS_SOURCE_FILE})
  add_dependencies(numeric generate_${ELEMENT})
endforeach()
