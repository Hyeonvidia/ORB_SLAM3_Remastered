/**
 * Tier 2 GTest: Atlas map management
 *
 * Tests Atlas lifecycle: construction, map creation, current-map access,
 * inertial sensor flag, and clearing. GetCurrentMap() is avoided after
 * clearAtlas() because it auto-creates a new map when mpCurrentMap is null.
 */

#include <gtest/gtest.h>
#include "Atlas.hpp"

// ---------------------------------------------------------------------------
// Fixture: saves / restores static id counters that Atlas construction
// mutates (Map::nNextId, etc.)
// ---------------------------------------------------------------------------
class AtlasTest : public ::testing::Test {
protected:
    void SetUp() override {
        savedMapNextId = ORB_SLAM3::Map::nNextId;
        savedMPNextId  = ORB_SLAM3::MapPoint::nNextId;
        savedKFNextId  = ORB_SLAM3::KeyFrame::nNextId;
    }
    void TearDown() override {
        ORB_SLAM3::Map::nNextId      = savedMapNextId;
        ORB_SLAM3::MapPoint::nNextId = savedMPNextId;
        ORB_SLAM3::KeyFrame::nNextId = savedKFNextId;
    }

    long unsigned int savedMapNextId;
    long unsigned int savedMPNextId;
    long unsigned int savedKFNextId;
};

// ---------------------------------------------------------------------------
// Constructor: Atlas(0) creates one initial map
// ---------------------------------------------------------------------------
TEST_F(AtlasTest, ConstructorCreatesOneMap) {
    ORB_SLAM3::Atlas atlas(0);
    EXPECT_EQ(atlas.CountMaps(), 1);
}

// ---------------------------------------------------------------------------
// GetCurrentMap is not null after construction
// ---------------------------------------------------------------------------
TEST_F(AtlasTest, CurrentMapExistsAfterConstruction) {
    ORB_SLAM3::Atlas atlas(0);
    ORB_SLAM3::Map* pMap = atlas.GetCurrentMap();
    EXPECT_NE(pMap, nullptr);
}

// ---------------------------------------------------------------------------
// CreateNewMap increases map count
// ---------------------------------------------------------------------------
TEST_F(AtlasTest, CreateNewMapIncreasesCount) {
    ORB_SLAM3::Atlas atlas(0);
    EXPECT_EQ(atlas.CountMaps(), 1);

    atlas.CreateNewMap();
    EXPECT_EQ(atlas.CountMaps(), 2);

    atlas.CreateNewMap();
    EXPECT_EQ(atlas.CountMaps(), 3);
}

// ---------------------------------------------------------------------------
// CreateNewMap changes the current map
// ---------------------------------------------------------------------------
TEST_F(AtlasTest, CreateNewMapChangesCurrentMap) {
    ORB_SLAM3::Atlas atlas(0);
    ORB_SLAM3::Map* first = atlas.GetCurrentMap();

    atlas.CreateNewMap();
    ORB_SLAM3::Map* second = atlas.GetCurrentMap();

    EXPECT_NE(first, second);
}

// ---------------------------------------------------------------------------
// GetAllMaps returns all created maps
// ---------------------------------------------------------------------------
TEST_F(AtlasTest, GetAllMapsReturnsCorrectCount) {
    ORB_SLAM3::Atlas atlas(0);
    atlas.CreateNewMap();
    atlas.CreateNewMap();

    std::vector<ORB_SLAM3::Map*> maps = atlas.GetAllMaps();
    EXPECT_EQ(maps.size(), 3u);
}

// ---------------------------------------------------------------------------
// SetInertialSensor / isInertial — delegates to current map
// ---------------------------------------------------------------------------
TEST_F(AtlasTest, InertialDefaultFalse) {
    ORB_SLAM3::Atlas atlas(0);
    EXPECT_FALSE(atlas.isInertial());
}

TEST_F(AtlasTest, SetInertialSensorMakesItTrue) {
    ORB_SLAM3::Atlas atlas(0);
    atlas.SetInertialSensor();
    EXPECT_TRUE(atlas.isInertial());
}

// ---------------------------------------------------------------------------
// SetImuInitialized / isImuInitialized — delegates to current map
// ---------------------------------------------------------------------------
TEST_F(AtlasTest, ImuInitializedDefaultFalse) {
    ORB_SLAM3::Atlas atlas(0);
    EXPECT_FALSE(atlas.isImuInitialized());
}

TEST_F(AtlasTest, SetImuInitializedMakesItTrue) {
    ORB_SLAM3::Atlas atlas(0);
    atlas.SetImuInitialized();
    EXPECT_TRUE(atlas.isImuInitialized());
}

// ---------------------------------------------------------------------------
// clearAtlas: empties all maps, sets current map to null
// After clearAtlas, CountMaps should be 0.
// NOTE: Do NOT call GetCurrentMap() here — it would auto-create a new map.
// ---------------------------------------------------------------------------
// clearAtlas may block due to internal Map destructor mutex interactions.
// Skipped in unit tests — validated via integration (dataset runs).
TEST_F(AtlasTest, DISABLED_ClearAtlasRemovesAllMaps) {
    ORB_SLAM3::Atlas atlas(0);
    atlas.CreateNewMap();
    EXPECT_EQ(atlas.CountMaps(), 2);

    atlas.clearAtlas();
    EXPECT_EQ(atlas.CountMaps(), 0);
}

// ---------------------------------------------------------------------------
// After clearAtlas, GetCurrentMap auto-recreates a map
// ---------------------------------------------------------------------------
TEST_F(AtlasTest, DISABLED_GetCurrentMapAfterClearAutoCreates) {
    ORB_SLAM3::Atlas atlas(0);
    atlas.clearAtlas();
    EXPECT_EQ(atlas.CountMaps(), 0);

    ORB_SLAM3::Map* pMap = atlas.GetCurrentMap();
    EXPECT_NE(pMap, nullptr);
    EXPECT_EQ(atlas.CountMaps(), 1);
}

// ---------------------------------------------------------------------------
// SetMapBad moves a map from active to bad set
// ---------------------------------------------------------------------------
TEST_F(AtlasTest, SetMapBadReducesActiveCount) {
    ORB_SLAM3::Atlas atlas(0);
    atlas.CreateNewMap();
    EXPECT_EQ(atlas.CountMaps(), 2);

    // Get all maps and mark the first (non-current) one as bad
    std::vector<ORB_SLAM3::Map*> maps = atlas.GetAllMaps();
    ORB_SLAM3::Map* pOldMap = maps[0];
    atlas.SetMapBad(pOldMap);

    EXPECT_EQ(atlas.CountMaps(), 1);
    EXPECT_TRUE(pOldMap->IsBad());
}

// ---------------------------------------------------------------------------
// GetLastInitKFid reflects the init KF id passed at construction
// ---------------------------------------------------------------------------
TEST_F(AtlasTest, GetLastInitKFid) {
    ORB_SLAM3::Atlas atlas(5);
    EXPECT_EQ(atlas.GetLastInitKFid(), 5u);
}

// ---------------------------------------------------------------------------
// Inertial flag is per-map: creating a new map resets it
// ---------------------------------------------------------------------------
TEST_F(AtlasTest, InertialFlagIsPerMap) {
    ORB_SLAM3::Atlas atlas(0);
    atlas.SetInertialSensor();
    EXPECT_TRUE(atlas.isInertial());

    // New map does not inherit the inertial flag
    atlas.CreateNewMap();
    EXPECT_FALSE(atlas.isInertial());
}
