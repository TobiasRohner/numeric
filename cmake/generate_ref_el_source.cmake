find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(NUMERIC_REF_EL_NAMES "ref_el_point;ref_el_segment;ref_el_tria;ref_el_quad;ref_el_tetra;ref_el_cube")


foreach (REF_EL ${NUMERIC_REF_EL_NAMES})
  set(REF_EL_SOURCE_FILE ${NUMERIC_INCLUDE_DIR}/numeric/mesh/${REF_EL}.hpp)
  list(APPEND NUMERIC_GENERATED_FILES ${REF_EL_SOURCE_FILE})
  execute_process(
    COMMAND ${Python3_EXECUTABLE} elements.py ${REF_EL}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts
    OUTPUT_VARIABLE ${REF_EL}_SRC
    COMMAND_ECHO STDOUT
  )
  file(
    GENERATE OUTPUT ${REF_EL_SOURCE_FILE}
    CONTENT "${${REF_EL}_SRC}"
  )
  add_custom_command(
    OUTPUT ${REF_EL_SOURCE_FILE}
    COMMAND ${Python3_EXECUTABLE} elements.py ${REF_EL} > ${REF_EL_SOURCE_FILE}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts
    DEPENDS ${CMAKE_SOURCE_DIR}/scripts/elements.py
    COMMENT Generating ${REF_EL_SOURCE_FILE}
  )
  add_custom_target(generate_${REF_EL} DEPENDS ${REF_EL_SOURCE_FILE})
  add_dependencies(numeric generate_${REF_EL})
endforeach()
