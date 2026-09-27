# Tests of the remaster's own classes that need no SLAM system around them.
find_package(Threads REQUIRED)

# The Reclaimer is compiled into the test, not linked from the library, so that
# it can be built a second time with ThreadSanitizer without instrumenting the
# library and everything under it.
foreach(variant "" "_tsan")
  set(t test_reclaimer${variant})
  add_executable(${t} test_reclaimer.cpp ${PROJECT_SOURCE_DIR}/src/atlas/Reclaimer.cpp)
  target_include_directories(${t} PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_compile_features(${t} PRIVATE cxx_std_17)
  target_link_libraries(${t} PRIVATE Threads::Threads)
  if(variant STREQUAL "_tsan")
    target_compile_options(${t} PRIVATE -fsanitize=thread -g -O1)
    target_link_options(${t} PRIVATE -fsanitize=thread)
  endif()
  add_test(NAME reclaimer${variant} COMMAND ${t})
  add_test(NAME reclaimer${variant}_dry_run COMMAND ${t} --dry-run)
  # A reader that keeps a pointer across its quiescent point without pinning
  # it, a check that skips the culled keyframes' slots, and one that does not
  # follow replaced-by links must each be caught: a test that cannot fail
  # proves nothing.
  foreach(broken forget_pins forget_culled forget_replaced)
    string(REPLACE "_" "-" flag ${broken})
    add_test(NAME reclaimer${variant}_${broken} COMMAND ${t} --${flag})
    set_tests_properties(reclaimer${variant}_${broken} PROPERTIES WILL_FAIL TRUE)
  endforeach()
endforeach()
