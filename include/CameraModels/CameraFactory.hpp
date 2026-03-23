#pragma once

#include "GeometricCamera.hpp"
#include "Pinhole.hpp"
#include "KannalaBrandt8.hpp"

#include <memory>
#include <vector>
#include <string>
#include <stdexcept>

namespace ORB_SLAM3 {

class CameraFactory {
public:
    enum class Type { Pinhole, KannalaBrandt8 };

    // Create camera from type enum and calibration parameters
    static GeometricCamera* create(Type type, const std::vector<float>& params) {
        switch (type) {
            case Type::Pinhole:
                return new Pinhole(params);
            case Type::KannalaBrandt8:
                return new KannalaBrandt8(params);
            default:
                throw std::runtime_error("CameraFactory: unknown camera type");
        }
    }

    // Create camera from string type name (for YAML config parsing)
    static GeometricCamera* create(const std::string& typeName, const std::vector<float>& params) {
        if (typeName == "PinHole" || typeName == "Rectified")
            return create(Type::Pinhole, params);
        else if (typeName == "KannalaBrandt8")
            return create(Type::KannalaBrandt8, params);
        else
            throw std::runtime_error("CameraFactory: unknown camera model '" + typeName + "'");
    }

    // Parse type string to enum
    static Type parseType(const std::string& typeName) {
        if (typeName == "PinHole" || typeName == "Rectified")
            return Type::Pinhole;
        else if (typeName == "KannalaBrandt8")
            return Type::KannalaBrandt8;
        else
            throw std::runtime_error("CameraFactory: unknown camera model '" + typeName + "'");
    }
};

} // namespace ORB_SLAM3
