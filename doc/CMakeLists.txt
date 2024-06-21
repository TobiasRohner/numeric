add_executable(generate_element_info_src generate_element_info_src.cpp)
target_link_libraries(generate_element_info_src PUBLIC numeric)

add_executable(generate_quad_rule_info_src generate_quad_rule_info_src.cpp)
target_link_libraries(generate_quad_rule_info_src PUBLIC numeric)

add_custom_target(
  doxygen_latex_src
  COMMAND ${CMAKE_CURRENT_BINARY_DIR}/generate_element_info_src && ${CMAKE_CURRENT_BINARY_DIR}/generate_quad_rule_info_src
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS generate_element_info_src generate_quad_rule_info_src
  COMMENT "Generating LaTeX Source for Documentation"
  VERBATIM
)

find_package(LATEX REQUIRED)
if (NUMERIC_DOXYGEN_QUIET)
  set(NUMERIC_LATEX_OUTPUT_SUPPRESSION 1> /dev/null)
else()
  set(NUMERIC_LATEX_OUTPUT_SUPPRESSION)
endif()
add_custom_target(
  doxygen_latex
  COMMAND find -maxdepth 1 -name *.tex -exec ${PDFLATEX_COMPILER} --shell-escape {} ${NUMERIC_LATEX_OUTPUT_SUPPRESSION} \;
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS doxygen_latex_src
  COMMENT "Generating Element Info Images"
  VERBATIM
)

find_package(Doxygen REQUIRED)
if (NUMERIC_DOXYGEN_QUIET)
  set(NUMERIC_DOXYGEN_QUIET_OPTION "YES")
else()
  set(NUMERIC_DOXYGEN_QUIET_OPTION "NO")
endif()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
add_custom_target(
  doxygen
  COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS doxygen_latex
  COMMENT "Generating API documentation with doxygen"
  VERBATIM
)
