import pytest
from unittest.mock import patch, MagicMock
from scidata.components.search import HybridSearchApp, BaseDocument


class TestBaseDocument:
    """Test BaseDocument dataclass"""

    def test_base_document_with_valid_id(self):
        """Test creating BaseDocument with valid id"""
        doc = BaseDocument(id="123")
        assert doc.id == "123"

    def test_base_document_with_none_id_raises_error(self):
        """Test that BaseDocument raises error if id is None"""
        with pytest.raises((ValueError, TypeError)):
            BaseDocument(id=None)


class TestHybridSearchApp:
    """Test HybridSearchApp class"""

    @patch('scidata.components.search.OpenSearch')
    @patch('scidata.components.search.OpenAI')
    def test_init(self, mock_openai, mock_opensearch):
        """Test HybridSearchApp initialization"""
        app = HybridSearchApp(
            index="test-index",
            field="content",
            model="text-embedding-3-small"
        )
        
        assert app.db_index == "test-index"
        assert app.field == "content"
        assert app.model == "text-embedding-3-small"
        assert app.opensearch_client is None
        assert app.openai_client is None

    @patch('scidata.components.search.OpenSearch')
    @patch('scidata.components.search.OpenAI')
    def test_context_manager_enter(self, mock_openai, mock_opensearch):
        """Test context manager __enter__ method"""
        mock_os_client = MagicMock()
        mock_opensearch.return_value = mock_os_client
        mock_openai.return_value = MagicMock()

        app = HybridSearchApp(
            index="test-index",
            field="content",
            model="text-embedding-3-small"
        )

        with app as context_app:
            assert context_app.opensearch_client is not None
            assert context_app.openai_client is not None

    @patch('scidata.components.search.OpenSearch')
    @patch('scidata.components.search.OpenAI')
    def test_normalize_embedding(self, mock_openai, mock_opensearch):
        """Test embedding normalization"""
        mock_os_client = MagicMock()
        mock_opensearch.return_value = mock_os_client
        mock_openai.return_value = MagicMock()

        app = HybridSearchApp(
            index="test-index",
            field="content",
            model="text-embedding-3-small"
        )

        with app:
            # Test normalization
            embedding = [3.0, 4.0]
            normalized = app.normalize(embedding)
            
            # L2 norm of [3, 4] is 5, so normalized should be [0.6, 0.8]
            assert abs(normalized[0] - 0.6) < 0.001
            assert abs(normalized[1] - 0.8) < 0.001

    @patch('scidata.components.search.OpenSearch')
    @patch('scidata.components.search.OpenAI')
    def test_normalize_embedding_zero_vector(self, mock_openai, mock_opensearch):
        """Test normalization of zero vector"""
        mock_os_client = MagicMock()
        mock_opensearch.return_value = mock_os_client
        mock_openai.return_value = MagicMock()

        app = HybridSearchApp(
            index="test-index",
            field="content",
            model="text-embedding-3-small"
        )

        with app:
            # Zero vector should return as-is
            zero_embedding = [0.0, 0.0, 0.0]
            normalized = app.normalize(zero_embedding)
            assert normalized == zero_embedding


class TestHybridSearchIntegration:
    """Integration tests for HybridSearchApp"""

    @patch('scidata.components.search.OpenSearch')
    @patch('scidata.components.search.OpenAI')
    def test_opensearch_index_body_structure(self, mock_openai, mock_opensearch):
        """Test that index body has correct structure"""
        index_body = HybridSearchApp.OPENSEARCH_INDEX_BODY
        
        assert "settings" in index_body
        assert "mappings" in index_body
        assert "index" in index_body["settings"]
        assert "properties" in index_body["mappings"]
        assert "embedding" in index_body["mappings"]["properties"]
        assert "content" in index_body["mappings"]["properties"]

    @patch('scidata.components.search.OpenSearch')
    @patch('scidata.components.search.OpenAI')
    def test_knn_settings_enabled(self, mock_openai, mock_opensearch):
        """Test that k-NN is enabled in index settings"""
        index_body = HybridSearchApp.OPENSEARCH_INDEX_BODY
        
        assert index_body["settings"]["index"]["knn"] is True
        assert index_body["settings"]["index"]["knn.algo_param.ef_search"] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
