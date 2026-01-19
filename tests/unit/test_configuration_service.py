"""Unit tests for configuration_service.py"""
import pytest
from backend.services.configuration_service import ConfigurationService

class TestConfigurationManagement:
    """Tests for configuration creation and updates"""
    
    def test_create_configuration(self, config_service, test_db):
        """Test creating a new configuration"""
        config = config_service.create_or_update_current_config(
            embedding_model="test-model",
            similarity_threshold=0.75,
            confidence_threshold=0.70
        )
        
        assert config is not None
        assert config.embedding_model == "test-model"
        assert config.similarity_threshold == 0.75
        assert config.is_current is True
    
    def test_update_configuration(self, config_service, test_db):
        """Test updating existing configuration"""
        config_service.create_or_update_current_config(
            embedding_model="model-1",
            similarity_threshold=0.75
        )
        
        updated = config_service.create_or_update_current_config(
            embedding_model="model-2",
            similarity_threshold=0.80
        )
        
        assert updated.embedding_model == "model-2"
        assert updated.similarity_threshold == 0.80
    
    def test_get_current_config(self, config_service, test_db):
        """Test getting current configuration"""
        config_service.create_or_update_current_config(
            embedding_model="test-model"
        )
        
        current = config_service.get_current_config()
        assert current is not None
        assert current.is_current is True

class TestVersionSnapshots:
    """Tests for configuration version snapshots"""
    
    def test_create_snapshot(self, config_service, test_db):
        """Test creating a configuration snapshot"""
        config_service.create_or_update_current_config(
            embedding_model="test-model",
            similarity_threshold=0.75
        )
        
        version = config_service.snapshot_configuration(
            version_label="v1.0",
            performance_metrics={"accuracy": 0.95}
        )
        
        assert version is not None
        assert version.version_label == "v1.0"
        assert version.embedding_model == "test-model"
        assert version.performance_metrics["accuracy"] == 0.95
    
    def test_get_all_versions(self, config_service, test_db):
        """Test getting all configuration versions"""
        config_service.create_or_update_current_config(embedding_model="model-1")
        config_service.snapshot_configuration(version_label="v1")
        config_service.snapshot_configuration(version_label="v2")
        
        versions = config_service.get_all_versions()
        assert len(versions) >= 2
    
    def test_get_version_by_id(self, config_service, test_db):
        """Test getting version by ID"""
        config_service.create_or_update_current_config(embedding_model="model-1")
        version = config_service.snapshot_configuration(version_label="v1")
        
        retrieved = config_service.get_version_by_id(version.id)
        assert retrieved is not None
        assert retrieved.id == version.id
        assert retrieved.version_label == "v1"

class TestKnownGoodVersions:
    """Tests for known-good version management"""
    
    def test_mark_known_good(self, config_service, test_db):
        """Test marking version as known-good"""
        config_service.create_or_update_current_config(embedding_model="model-1")
        version = config_service.snapshot_configuration(version_label="v1")
        
        marked = config_service.mark_version_as_known_good(version.id)
        assert marked is not None
        assert marked.is_known_good is True
    
    def test_get_known_good_versions(self, config_service, test_db):
        """Test getting known-good versions"""
        config_service.create_or_update_current_config(embedding_model="model-1")
        v1 = config_service.snapshot_configuration(version_label="v1")
        v2 = config_service.snapshot_configuration(version_label="v2")
        
        config_service.mark_version_as_known_good(v1.id)
        
        known_good = config_service.get_known_good_versions()
        assert len(known_good) >= 1
        assert any(v.id == v1.id for v in known_good)

class TestRollback:
    """Tests for configuration rollback"""
    
    def test_restore_configuration(self, config_service, test_db):
        """Test restoring configuration from version"""
        config_service.create_or_update_current_config(
            embedding_model="model-1",
            similarity_threshold=0.75
        )
        version = config_service.snapshot_configuration(version_label="v1")
        
        # Change current config
        config_service.create_or_update_current_config(
            embedding_model="model-2",
            similarity_threshold=0.80
        )
        
        # Restore from version
        restored = config_service.restore_configuration_from_version(version.id)
        assert restored.embedding_model == "model-1"
        assert restored.similarity_threshold == 0.75
    
    def test_restore_nonexistent_version(self, config_service, test_db):
        """Test restoring from non-existent version raises error"""
        with pytest.raises(ValueError):
            config_service.restore_configuration_from_version(99999)

class TestBestVersionSelection:
    """Tests for best version selection by metrics"""
    
    def test_get_best_version(self, config_service, test_db):
        """Test getting best version by metrics"""
        config_service.create_or_update_current_config(embedding_model="model-1")
        
        v1 = config_service.snapshot_configuration(
            version_label="v1",
            performance_metrics={"accuracy": 0.90, "error_rate": 0.05}
        )
        v2 = config_service.snapshot_configuration(
            version_label="v2",
            performance_metrics={"accuracy": 0.95, "error_rate": 0.02}
        )
        
        config_service.mark_version_as_known_good(v1.id)
        config_service.mark_version_as_known_good(v2.id)
        
        best = config_service.get_best_version_by_metrics()
        assert best is not None
        # Should prefer v2 with higher accuracy and lower error rate
        assert best.id in [v1.id, v2.id]
