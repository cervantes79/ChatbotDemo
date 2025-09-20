"""
Comprehensive Test Cases for Prompt Enhancement Benchmark
100+ test cases across different categories
Author: Barış Genç
"""

from typing import List, Dict

class TestCasesDataset:
    """Comprehensive dataset of 100+ test cases for prompt enhancement evaluation"""

    def __init__(self):
        self.test_cases = self._generate_test_cases()

    def _generate_test_cases(self) -> List[Dict]:
        """Generate comprehensive test cases across multiple categories"""

        business_cases = [
            {"query": "What are the company work hours?", "category": "business", "expected_concepts": ["work", "company", "hours"]},
            {"query": "How do I request vacation time?", "category": "business", "expected_concepts": ["vacation", "employee", "procedure"]},
            {"query": "What is the employee dress code policy?", "category": "business", "expected_concepts": ["employee", "policy", "dress"]},
            {"query": "When is the next team meeting scheduled?", "category": "business", "expected_concepts": ["team", "meeting", "schedule"]},
            {"query": "What are the salary review procedures?", "category": "business", "expected_concepts": ["salary", "procedure", "review"]},
            {"query": "How do I submit expense reports?", "category": "business", "expected_concepts": ["expense", "procedure", "reports"]},
            {"query": "What benefits does the company offer?", "category": "business", "expected_concepts": ["benefits", "company", "offer"]},
            {"query": "Who is my direct manager?", "category": "business", "expected_concepts": ["manager", "employee", "direct"]},
            {"query": "What is the remote work policy?", "category": "business", "expected_concepts": ["remote", "work", "policy"]},
            {"query": "How do I book a conference room?", "category": "business", "expected_concepts": ["conference", "room", "booking"]},
            {"query": "What is the company mission statement?", "category": "business", "expected_concepts": ["company", "mission", "statement"]},
            {"query": "How do I access the employee portal?", "category": "business", "expected_concepts": ["employee", "portal", "access"]},
            {"query": "What is the deadline for project submissions?", "category": "business", "expected_concepts": ["deadline", "project", "submission"]},
            {"query": "How do I report a workplace incident?", "category": "business", "expected_concepts": ["report", "workplace", "incident"]},
            {"query": "What training programs are available?", "category": "business", "expected_concepts": ["training", "programs", "available"]},
            {"query": "How do I change my direct deposit information?", "category": "business", "expected_concepts": ["change", "deposit", "information"]},
            {"query": "What is the company's diversity policy?", "category": "business", "expected_concepts": ["company", "diversity", "policy"]},
            {"query": "How do I schedule a performance review?", "category": "business", "expected_concepts": ["schedule", "performance", "review"]},
            {"query": "What are the IT security guidelines?", "category": "business", "expected_concepts": ["security", "guidelines", "policy"]},
            {"query": "How do I request office supplies?", "category": "business", "expected_concepts": ["request", "office", "supplies"]},
            {"query": "What is the employee referral program?", "category": "business", "expected_concepts": ["employee", "referral", "program"]},
            {"query": "How do I update my emergency contact?", "category": "business", "expected_concepts": ["update", "emergency", "contact"]},
            {"query": "What are the parking regulations?", "category": "business", "expected_concepts": ["parking", "regulations", "policy"]},
            {"query": "How do I join the company health plan?", "category": "business", "expected_concepts": ["health", "plan", "benefits"]},
            {"query": "What is the procedure for reporting bugs?", "category": "business", "expected_concepts": ["procedure", "reporting", "bugs"]},
        ]

        technical_cases = [
            {"query": "How do I optimize database performance?", "category": "technical", "expected_concepts": ["database", "performance", "optimization"]},
            {"query": "What is the best sorting algorithm for large datasets?", "category": "technical", "expected_concepts": ["algorithm", "sorting", "data"]},
            {"query": "How do I implement user authentication?", "category": "technical", "expected_concepts": ["authentication", "security", "system"]},
            {"query": "What are the best practices for API design?", "category": "technical", "expected_concepts": ["API", "design", "best"]},
            {"query": "How do I debug memory leaks in applications?", "category": "technical", "expected_concepts": ["debug", "memory", "application"]},
            {"query": "What is the difference between SQL and NoSQL?", "category": "technical", "expected_concepts": ["database", "SQL", "NoSQL"]},
            {"query": "How do I set up continuous integration?", "category": "technical", "expected_concepts": ["integration", "deployment", "system"]},
            {"query": "What are microservices architecture benefits?", "category": "technical", "expected_concepts": ["microservices", "architecture", "system"]},
            {"query": "How do I implement caching strategies?", "category": "technical", "expected_concepts": ["caching", "performance", "strategy"]},
            {"query": "What is test-driven development?", "category": "technical", "expected_concepts": ["testing", "development", "methodology"]},
            {"query": "How do I secure web applications?", "category": "technical", "expected_concepts": ["security", "web", "application"]},
            {"query": "What are design patterns in programming?", "category": "technical", "expected_concepts": ["design", "patterns", "programming"]},
            {"query": "How do I manage version control conflicts?", "category": "technical", "expected_concepts": ["version", "control", "conflicts"]},
            {"query": "What is machine learning model deployment?", "category": "technical", "expected_concepts": ["machine", "learning", "deployment"]},
            {"query": "How do I optimize frontend performance?", "category": "technical", "expected_concepts": ["frontend", "performance", "optimization"]},
            {"query": "What are containerization benefits?", "category": "technical", "expected_concepts": ["containerization", "deployment", "benefits"]},
            {"query": "How do I implement load balancing?", "category": "technical", "expected_concepts": ["load", "balancing", "system"]},
            {"query": "What is the purpose of unit testing?", "category": "technical", "expected_concepts": ["unit", "testing", "quality"]},
            {"query": "How do I handle concurrent programming?", "category": "technical", "expected_concepts": ["concurrent", "programming", "threading"]},
            {"query": "What are blockchain fundamentals?", "category": "technical", "expected_concepts": ["blockchain", "fundamentals", "technology"]},
            {"query": "How do I implement data validation?", "category": "technical", "expected_concepts": ["data", "validation", "security"]},
            {"query": "What is cloud computing architecture?", "category": "technical", "expected_concepts": ["cloud", "computing", "architecture"]},
            {"query": "How do I optimize mobile app performance?", "category": "technical", "expected_concepts": ["mobile", "app", "performance"]},
            {"query": "What are cybersecurity best practices?", "category": "technical", "expected_concepts": ["cybersecurity", "security", "practices"]},
            {"query": "How do I implement real-time notifications?", "category": "technical", "expected_concepts": ["real-time", "notifications", "system"]},
        ]

        weather_cases = [
            {"query": "What's the weather like in Istanbul?", "category": "weather", "expected_concepts": ["weather", "Istanbul", "location"]},
            {"query": "Will it rain tomorrow in London?", "category": "weather", "expected_concepts": ["rain", "weather", "forecast"]},
            {"query": "What's the temperature in New York?", "category": "weather", "expected_concepts": ["temperature", "weather", "location"]},
            {"query": "Is it going to be sunny this weekend?", "category": "weather", "expected_concepts": ["sunny", "weather", "forecast"]},
            {"query": "How's the weather in Tokyo today?", "category": "weather", "expected_concepts": ["weather", "Tokyo", "today"]},
            {"query": "What's the forecast for Paris this week?", "category": "weather", "expected_concepts": ["forecast", "weather", "Paris"]},
            {"query": "Will there be snow in Moscow?", "category": "weather", "expected_concepts": ["snow", "weather", "Moscow"]},
            {"query": "What's the humidity level in Miami?", "category": "weather", "expected_concepts": ["humidity", "weather", "Miami"]},
            {"query": "Is it windy in Chicago today?", "category": "weather", "expected_concepts": ["windy", "weather", "Chicago"]},
            {"query": "What's the weather forecast for Berlin?", "category": "weather", "expected_concepts": ["weather", "forecast", "Berlin"]},
            {"query": "Will it be hot in Dubai tomorrow?", "category": "weather", "expected_concepts": ["hot", "weather", "Dubai"]},
            {"query": "What's the current temperature in Sydney?", "category": "weather", "expected_concepts": ["temperature", "weather", "Sydney"]},
            {"query": "Is there a storm coming to Florida?", "category": "weather", "expected_concepts": ["storm", "weather", "Florida"]},
            {"query": "What's the weather like in Barcelona?", "category": "weather", "expected_concepts": ["weather", "Barcelona", "location"]},
            {"query": "Will it be cloudy in Vancouver?", "category": "weather", "expected_concepts": ["cloudy", "weather", "Vancouver"]},
            {"query": "What's the weather condition in Mumbai?", "category": "weather", "expected_concepts": ["weather", "condition", "Mumbai"]},
            {"query": "Is it foggy in San Francisco?", "category": "weather", "expected_concepts": ["foggy", "weather", "San Francisco"]},
            {"query": "What's the weather report for Cairo?", "category": "weather", "expected_concepts": ["weather", "report", "Cairo"]},
            {"query": "Will there be thunderstorms in Atlanta?", "category": "weather", "expected_concepts": ["thunderstorms", "weather", "Atlanta"]},
            {"query": "What's the air pressure in Denver?", "category": "weather", "expected_concepts": ["pressure", "weather", "Denver"]},
        ]

        general_knowledge_cases = [
            {"query": "What is the capital of France?", "category": "general", "expected_concepts": ["capital", "France", "location"]},
            {"query": "Who invented the telephone?", "category": "general", "expected_concepts": ["invented", "telephone", "history"]},
            {"query": "How do photosynthesis work?", "category": "general", "expected_concepts": ["photosynthesis", "biology", "process"]},
            {"query": "What is the largest ocean on Earth?", "category": "general", "expected_concepts": ["largest", "ocean", "Earth"]},
            {"query": "When did World War II end?", "category": "general", "expected_concepts": ["World War", "end", "history"]},
            {"query": "What is the speed of light?", "category": "general", "expected_concepts": ["speed", "light", "physics"]},
            {"query": "Who wrote Romeo and Juliet?", "category": "general", "expected_concepts": ["wrote", "Romeo", "literature"]},
            {"query": "What is the formula for water?", "category": "general", "expected_concepts": ["formula", "water", "chemistry"]},
            {"query": "How many continents are there?", "category": "general", "expected_concepts": ["continents", "geography", "number"]},
            {"query": "What is the smallest planet?", "category": "general", "expected_concepts": ["smallest", "planet", "astronomy"]},
            {"query": "Who painted the Mona Lisa?", "category": "general", "expected_concepts": ["painted", "Mona Lisa", "art"]},
            {"query": "What is democracy?", "category": "general", "expected_concepts": ["democracy", "government", "political"]},
            {"query": "How do vaccines work?", "category": "general", "expected_concepts": ["vaccines", "medical", "immunology"]},
            {"query": "What causes earthquakes?", "category": "general", "expected_concepts": ["earthquakes", "geology", "causes"]},
            {"query": "Who discovered penicillin?", "category": "general", "expected_concepts": ["discovered", "penicillin", "medical"]},
        ]

        complex_reasoning_cases = [
            {"query": "How would artificial intelligence impact future job markets?", "category": "complex", "expected_concepts": ["artificial intelligence", "job", "future"]},
            {"query": "What are the ethical implications of genetic engineering?", "category": "complex", "expected_concepts": ["ethical", "genetic", "engineering"]},
            {"query": "How can we solve climate change effectively?", "category": "complex", "expected_concepts": ["climate", "change", "solution"]},
            {"query": "What strategies can reduce income inequality?", "category": "complex", "expected_concepts": ["income", "inequality", "strategies"]},
            {"query": "How do economic policies affect social welfare?", "category": "complex", "expected_concepts": ["economic", "policies", "welfare"]},
            {"query": "What are the long-term effects of remote work?", "category": "complex", "expected_concepts": ["remote", "work", "effects"]},
            {"query": "How can education systems adapt to digital transformation?", "category": "complex", "expected_concepts": ["education", "digital", "transformation"]},
            {"query": "What role does technology play in mental health?", "category": "complex", "expected_concepts": ["technology", "mental", "health"]},
            {"query": "How can cities become more sustainable?", "category": "complex", "expected_concepts": ["cities", "sustainable", "environment"]},
            {"query": "What are the implications of space exploration for humanity?", "category": "complex", "expected_concepts": ["space", "exploration", "humanity"]},
            {"query": "How do cultural differences affect global business?", "category": "complex", "expected_concepts": ["cultural", "global", "business"]},
            {"query": "What strategies can combat misinformation online?", "category": "complex", "expected_concepts": ["misinformation", "online", "strategies"]},
            {"query": "How can renewable energy replace fossil fuels?", "category": "complex", "expected_concepts": ["renewable", "energy", "fossil"]},
            {"query": "What are the societal impacts of social media?", "category": "complex", "expected_concepts": ["societal", "social media", "impacts"]},
            {"query": "How can healthcare systems prepare for future pandemics?", "category": "complex", "expected_concepts": ["healthcare", "pandemics", "preparation"]},
        ]

        # Combine all test cases
        all_cases = business_cases + technical_cases + weather_cases + general_knowledge_cases + complex_reasoning_cases

        # Add metadata to each case
        for i, case in enumerate(all_cases):
            case["id"] = f"test_{i+1:03d}"
            case["difficulty"] = self._assess_difficulty(case["query"])

        return all_cases

    def _assess_difficulty(self, query: str) -> str:
        """Assess the difficulty level of a query"""
        if any(word in query.lower() for word in ["how", "why", "explain", "strategy", "impact", "implication"]):
            return "hard"
        elif any(word in query.lower() for word in ["what", "when", "where", "who"]):
            return "easy"
        else:
            return "medium"

    def get_test_cases(self) -> List[Dict]:
        """Get all test cases"""
        return self.test_cases

    def get_test_cases_by_category(self, category: str) -> List[Dict]:
        """Get test cases filtered by category"""
        return [case for case in self.test_cases if case["category"] == category]

    def get_test_cases_by_difficulty(self, difficulty: str) -> List[Dict]:
        """Get test cases filtered by difficulty"""
        return [case for case in self.test_cases if case["difficulty"] == difficulty]

    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        categories = {}
        difficulties = {}

        for case in self.test_cases:
            cat = case["category"]
            diff = case["difficulty"]

            categories[cat] = categories.get(cat, 0) + 1
            difficulties[diff] = difficulties.get(diff, 0) + 1

        return {
            "total_cases": len(self.test_cases),
            "categories": categories,
            "difficulties": difficulties
        }