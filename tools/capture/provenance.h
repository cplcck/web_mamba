#pragma once
#include "artifact.h"
namespace capture {
json provenance(const fs::path & manifest_path);
Fixture supplied_fixture(const fs::path & path, const std::string & selected, const json & manifest);
}
