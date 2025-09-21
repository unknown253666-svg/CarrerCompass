import sys
import os
import unittest

# Add the parent directory to the path so we can import from backend
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from CareerCompass.backend.scoring import calculate_final_score, generate_feedback

class TestScoring(unittest.TestCase):
    """Test cases for scoring system"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.sample_resume = """
        John Doe
        Software Engineer
        Experienced Python developer with skills in data analysis and machine learning.
        Proficient in SQL, Pandas, and data visualization tools.
        """
        
        self.sample_jd = """
        Senior Data Scientist
        We are looking for a candidate with experience in Python, machine learning, and data analysis.
        Required skills include Spark, Kafka, and deep learning.
        Experience with cloud platforms like AWS or GCP is a plus.
        """
    
    def test_calculate_final_score(self):
        """Test that calculate_final_score returns expected structure"""
        result = calculate_final_score(self.sample_resume, self.sample_jd)
        
        # Check that all expected keys are present
        self.assertIn("hard_score", result)
        self.assertIn("semantic_score", result)
        self.assertIn("total_score", result)
        self.assertIn("verdict", result)
        self.assertIn("missing_skills", result)
        
        # Check that scores are in expected range
        self.assertGreaterEqual(result["hard_score"], 0)
        self.assertLessEqual(result["hard_score"], 100)
        self.assertGreaterEqual(result["semantic_score"], 0)
        self.assertLessEqual(result["semantic_score"], 100)
        self.assertGreaterEqual(result["total_score"], 0)
        self.assertLessEqual(result["total_score"], 100)
        
        # Check that verdict is one of the expected values
        self.assertIn(result["verdict"], ["High", "Medium", "Low"])
        
        # Check that missing_skills is a list
        self.assertIsInstance(result["missing_skills"], list)
    
    def test_generate_feedback(self):
        """Test that generate_feedback returns a string"""
        score_data = calculate_final_score(self.sample_resume, self.sample_jd)
        feedback = generate_feedback(self.sample_resume, self.sample_jd, score_data)
        self.assertIsInstance(feedback, str)

if __name__ == '__main__':
    unittest.main()