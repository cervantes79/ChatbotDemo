"""
1000+ Comprehensive Test Dataset Generator
For detailed ICAR vs No-ICAR comparison
Author: Barış Genç
"""

import json
import random
from typing import List, Dict

class MassiveTestDataset:
    """Generate 1000+ diverse test prompts for comprehensive testing"""

    def __init__(self):
        self.test_prompts = []

    def generate_business_prompts(self, count: int = 200) -> List[Dict]:
        """Generate business-related prompts"""
        templates = [
            "What are the {item} for {context}?",
            "How do I {action} {object}?",
            "What is the {item} policy?",
            "When is the {event} scheduled?",
            "Who is responsible for {task}?",
            "What are the requirements for {process}?",
            "How can I access {resource}?",
            "What is the procedure for {action}?",
            "What benefits does {entity} offer?",
            "How do I report {issue}?"
        ]

        items = ["work hours", "dress code", "vacation policy", "salary review", "training program",
                "parking regulations", "health plan", "employee handbook", "IT security guidelines",
                "expense procedures", "meeting schedule", "office supplies", "remote work policy"]

        contexts = ["new employees", "contractors", "remote workers", "full-time staff", "managers",
                   "part-time workers", "interns", "consultants", "executives", "team leads"]

        actions = ["request", "submit", "schedule", "update", "change", "cancel", "approve",
                  "review", "access", "complete", "report", "track", "manage", "organize"]

        objects = ["vacation time", "expense reports", "performance review", "team meeting",
                  "conference room", "parking spot", "equipment", "software access", "training",
                  "overtime approval", "budget allocation", "project timeline"]

        events = ["next board meeting", "quarterly review", "annual conference", "team building",
                 "training session", "budget planning", "project deadline", "performance evaluation"]

        tasks = ["payroll processing", "inventory management", "customer support", "data backup",
                "security monitoring", "compliance auditing", "quality assurance", "risk assessment"]

        processes = ["hiring", "onboarding", "promotion", "termination", "project approval",
                    "budget allocation", "equipment procurement", "system maintenance"]

        resources = ["employee portal", "company intranet", "shared drives", "training materials",
                    "IT support", "HR database", "financial reports", "project management tools"]

        issues = ["workplace incident", "security breach", "equipment failure", "policy violation",
                 "harassment complaint", "safety concern", "technical issue", "budget overrun"]

        entities = ["the company", "HR department", "IT support", "management", "our organization",
                   "the benefits program", "employee assistance", "training department"]

        prompts = []
        for i in range(count):
            template = random.choice(templates)
            prompt = template.format(
                item=random.choice(items),
                context=random.choice(contexts),
                action=random.choice(actions),
                object=random.choice(objects),
                event=random.choice(events),
                task=random.choice(tasks),
                process=random.choice(processes),
                resource=random.choice(resources),
                issue=random.choice(issues),
                entity=random.choice(entities)
            )

            prompts.append({
                "id": f"business_{i+1:03d}",
                "category": "business",
                "prompt": prompt,
                "expected_concepts": self._extract_concepts(prompt, "business")
            })

        return prompts

    def generate_technical_prompts(self, count: int = 200) -> List[Dict]:
        """Generate technical prompts"""
        templates = [
            "How do I {action} {technology}?",
            "What is the best way to {optimize} {system}?",
            "How can I {implement} {feature}?",
            "What are the {benefits} of {approach}?",
            "How do I {debug} {problem}?",
            "What is {concept} and how does it work?",
            "How can I {secure} {application}?",
            "What tools are best for {task}?",
            "How do I {deploy} {system}?",
            "What are common {issues} with {technology}?"
        ]

        actions = ["install", "configure", "optimize", "troubleshoot", "upgrade", "migrate",
                  "integrate", "automate", "monitor", "backup", "restore", "scale"]

        technologies = ["database", "web server", "API", "microservices", "container", "cloud infrastructure",
                       "machine learning model", "blockchain", "mobile app", "web application", "neural network"]

        optimizes = ["improve performance of", "scale", "secure", "optimize", "refactor", "modernize",
                    "streamline", "enhance", "accelerate", "stabilize"]

        systems = ["database system", "web application", "API gateway", "load balancer", "caching layer",
                  "monitoring system", "CI/CD pipeline", "authentication system", "payment system"]

        implements = ["implement", "develop", "create", "build", "design", "architect", "deploy"]

        features = ["user authentication", "real-time notifications", "payment processing", "data analytics",
                   "search functionality", "caching mechanism", "logging system", "error handling"]

        benefits = ["advantages", "benefits", "pros and cons", "use cases", "performance implications"]

        approaches = ["microservices architecture", "serverless computing", "event-driven architecture",
                     "REST API design", "GraphQL", "containerization", "test-driven development"]

        debugs = ["debug", "troubleshoot", "fix", "resolve", "diagnose", "identify"]

        problems = ["memory leaks", "performance issues", "connection timeouts", "authentication errors",
                   "data corruption", "race conditions", "deadlocks", "security vulnerabilities"]

        concepts = ["machine learning", "blockchain", "microservices", "containerization", "DevOps",
                   "artificial intelligence", "cloud computing", "cybersecurity", "data science"]

        secures = ["secure", "protect", "harden", "encrypt", "authenticate"]

        applications = ["web application", "mobile app", "API", "database", "cloud infrastructure",
                       "IoT system", "payment gateway", "user data", "network traffic"]

        tasks = ["code review", "testing", "deployment", "monitoring", "backup", "documentation",
                "performance testing", "security auditing", "data migration", "system integration"]

        deploys = ["deploy", "release", "publish", "launch", "rollout", "distribute"]

        issues = ["security vulnerabilities", "performance bottlenecks", "scalability issues", "bugs",
                 "compatibility problems", "integration challenges", "maintenance issues"]

        prompts = []
        for i in range(count):
            template = random.choice(templates)
            prompt = template.format(
                action=random.choice(actions),
                technology=random.choice(technologies),
                optimize=random.choice(optimizes),
                system=random.choice(systems),
                implement=random.choice(implements),
                feature=random.choice(features),
                benefits=random.choice(benefits),
                approach=random.choice(approaches),
                debug=random.choice(debugs),
                problem=random.choice(problems),
                concept=random.choice(concepts),
                secure=random.choice(secures),
                application=random.choice(applications),
                task=random.choice(tasks),
                deploy=random.choice(deploys),
                issues=random.choice(issues)
            )

            prompts.append({
                "id": f"technical_{i+1:03d}",
                "category": "technical",
                "prompt": prompt,
                "expected_concepts": self._extract_concepts(prompt, "technical")
            })

        return prompts

    def generate_general_prompts(self, count: int = 200) -> List[Dict]:
        """Generate general knowledge prompts"""
        templates = [
            "What is {concept}?",
            "How does {process} work?",
            "What are the {aspects} of {topic}?",
            "Why is {phenomenon} important?",
            "What causes {event}?",
            "How can I {action} {object}?",
            "What is the difference between {item1} and {item2}?",
            "What are the benefits of {activity}?",
            "How do I {learn} about {subject}?",
            "What factors affect {outcome}?"
        ]

        concepts = ["artificial intelligence", "climate change", "democracy", "evolution", "quantum physics",
                   "photosynthesis", "economics", "psychology", "philosophy", "literature", "art", "music"]

        processes = ["photosynthesis", "digestion", "cellular respiration", "water cycle", "plate tectonics",
                    "immune system", "nervous system", "reproduction", "genetic inheritance"]

        aspects = ["key features", "main components", "important factors", "essential elements", "core principles"]

        topics = ["environmental science", "human behavior", "cultural diversity", "technological advancement",
                 "social media", "education", "healthcare", "sustainable development", "space exploration"]

        phenomena = ["biodiversity", "cultural exchange", "scientific research", "artistic expression",
                    "technological innovation", "social cooperation", "environmental conservation"]

        events = ["earthquakes", "volcanic eruptions", "economic recession", "climate change", "pandemics",
                 "social movements", "technological disruption", "cultural shifts", "political changes"]

        actions = ["learn", "understand", "study", "research", "explore", "investigate", "analyze", "evaluate"]

        objects = ["new language", "musical instrument", "scientific concept", "historical event", "cultural tradition",
                  "technical skill", "artistic technique", "philosophical idea", "mathematical principle"]

        items1 = ["renewable energy", "fossil fuels", "democracy", "autocracy", "bacteria", "viruses",
                 "classical music", "modern music", "traditional art", "contemporary art"]

        items2 = ["solar power", "wind power", "republic", "monarchy", "fungi", "algae",
                 "jazz music", "rock music", "digital art", "sculpture"]

        activities = ["exercise", "meditation", "reading", "traveling", "volunteering", "recycling",
                     "learning languages", "playing music", "gardening", "cooking", "writing"]

        learns = ["learn", "study", "research", "understand", "explore", "discover"]

        subjects = ["history", "science", "mathematics", "literature", "philosophy", "psychology",
                   "sociology", "anthropology", "linguistics", "archaeology", "astronomy"]

        outcomes = ["health", "happiness", "success", "creativity", "productivity", "longevity",
                   "intelligence", "memory", "decision-making", "problem-solving"]

        prompts = []
        for i in range(count):
            template = random.choice(templates)
            prompt = template.format(
                concept=random.choice(concepts),
                process=random.choice(processes),
                aspects=random.choice(aspects),
                topic=random.choice(topics),
                phenomenon=random.choice(phenomena),
                event=random.choice(events),
                action=random.choice(actions),
                object=random.choice(objects),
                item1=random.choice(items1),
                item2=random.choice(items2),
                activity=random.choice(activities),
                learn=random.choice(learns),
                subject=random.choice(subjects),
                outcome=random.choice(outcomes)
            )

            prompts.append({
                "id": f"general_{i+1:03d}",
                "category": "general",
                "prompt": prompt,
                "expected_concepts": self._extract_concepts(prompt, "general")
            })

        return prompts

    def generate_complex_prompts(self, count: int = 200) -> List[Dict]:
        """Generate complex reasoning prompts"""
        templates = [
            "How would {factor1} impact {factor2} in the context of {context}?",
            "What are the ethical implications of {technology} for {stakeholder}?",
            "How can {solution} address {problem} while considering {constraint}?",
            "What strategies would be most effective for {goal} given {limitation}?",
            "How do {element1} and {element2} interact to influence {outcome}?",
            "What would be the long-term consequences of {action} on {system}?",
            "How can organizations balance {priority1} with {priority2}?",
            "What role does {factor} play in determining {result}?",
            "How might {trend} reshape {industry} over the next decade?",
            "What are the trade-offs between {approach1} and {approach2} for {purpose}?"
        ]

        factor1s = ["artificial intelligence", "climate change", "automation", "globalization", "urbanization",
                   "demographic shifts", "technological disruption", "economic inequality", "cultural diversity"]

        factor2s = ["employment patterns", "social structures", "educational systems", "healthcare delivery",
                   "political governance", "environmental sustainability", "economic development", "cultural identity"]

        contexts = ["developing countries", "urban environments", "rural communities", "digital economies",
                   "aging populations", "multicultural societies", "post-pandemic world", "sustainable development"]

        technologies = ["genetic engineering", "autonomous vehicles", "facial recognition", "artificial intelligence",
                       "quantum computing", "brain-computer interfaces", "nanotechnology", "virtual reality"]

        stakeholders = ["society", "individuals", "workers", "students", "elderly population", "minorities",
                       "future generations", "developing nations", "small businesses", "consumers"]

        solutions = ["renewable energy", "universal basic income", "remote education", "telemedicine",
                    "sustainable agriculture", "circular economy", "digital governance", "community-based programs"]

        problems = ["climate change", "poverty", "inequality", "unemployment", "healthcare access",
                   "educational gaps", "social isolation", "environmental degradation", "food insecurity"]

        constraints = ["limited resources", "political resistance", "technological barriers", "cultural differences",
                      "economic costs", "time limitations", "regulatory challenges", "implementation complexity"]

        goals = ["achieving sustainability", "reducing inequality", "improving education", "enhancing healthcare",
                "promoting innovation", "building resilience", "fostering inclusion", "ensuring security"]

        limitations = ["budget constraints", "technological readiness", "regulatory frameworks", "public acceptance",
                      "infrastructure requirements", "skill gaps", "cultural barriers", "political instability"]

        elements = ["economic policies", "social programs", "technological innovation", "cultural factors",
                   "environmental conditions", "political systems", "educational initiatives", "healthcare systems"]

        outcomes = ["social mobility", "economic growth", "environmental health", "public wellbeing",
                   "innovation capacity", "social cohesion", "cultural preservation", "sustainable development"]

        actions = ["implementing universal healthcare", "adopting renewable energy", "automating manufacturing",
                  "digitizing education", "restructuring economies", "regulating AI", "promoting diversity"]

        systems = ["global economy", "social fabric", "political landscape", "environmental ecosystems",
                  "educational institutions", "healthcare systems", "technological infrastructure"]

        priorities = ["economic growth", "environmental protection", "social equity", "individual privacy",
                     "national security", "innovation", "cultural preservation", "global cooperation"]

        factors = ["education", "technology", "culture", "economics", "politics", "environment",
                  "demographics", "social networks", "institutional frameworks", "historical context"]

        results = ["social outcomes", "economic performance", "environmental impact", "political stability",
                  "cultural vitality", "technological advancement", "human wellbeing", "sustainable development"]

        trends = ["remote work", "artificial intelligence", "sustainable consumption", "digital transformation",
                 "aging populations", "urbanization", "climate adaptation", "personalized medicine"]

        industries = ["healthcare", "education", "manufacturing", "agriculture", "transportation", "energy",
                     "entertainment", "financial services", "retail", "telecommunications"]

        approaches = ["top-down regulation", "market-based solutions", "community-driven initiatives",
                     "technological innovation", "international cooperation", "public-private partnerships"]

        purposes = ["addressing inequality", "promoting sustainability", "enhancing security", "fostering innovation",
                   "improving efficiency", "building resilience", "ensuring accessibility", "maintaining quality"]

        prompts = []
        for i in range(count):
            template = random.choice(templates)
            prompt = template.format(
                factor1=random.choice(factor1s),
                factor2=random.choice(factor2s),
                context=random.choice(contexts),
                technology=random.choice(technologies),
                stakeholder=random.choice(stakeholders),
                solution=random.choice(solutions),
                problem=random.choice(problems),
                constraint=random.choice(constraints),
                goal=random.choice(goals),
                limitation=random.choice(limitations),
                element1=random.choice(elements),
                element2=random.choice(elements),
                outcome=random.choice(outcomes),
                action=random.choice(actions),
                system=random.choice(systems),
                priority1=random.choice(priorities),
                priority2=random.choice(priorities),
                factor=random.choice(factors),
                result=random.choice(results),
                trend=random.choice(trends),
                industry=random.choice(industries),
                approach1=random.choice(approaches),
                approach2=random.choice(approaches),
                purpose=random.choice(purposes)
            )

            prompts.append({
                "id": f"complex_{i+1:03d}",
                "category": "complex",
                "prompt": prompt,
                "expected_concepts": self._extract_concepts(prompt, "complex")
            })

        return prompts

    def generate_creative_prompts(self, count: int = 200) -> List[Dict]:
        """Generate creative and open-ended prompts"""
        templates = [
            "Write a {creative_type} about {theme}",
            "Explain {concept} as if you were {character}",
            "Create a {format} that demonstrates {principle}",
            "Imagine {scenario} and describe {aspect}",
            "Design a {system} for {purpose} in {setting}",
            "Tell a story where {element1} meets {element2}",
            "Create an analogy between {item1} and {item2}",
            "Describe {experience} from the perspective of {viewpoint}",
            "Generate ideas for {project} that incorporates {theme}",
            "Explain how {future_tech} might change {domain}"
        ]

        creative_types = ["short story", "poem", "dialogue", "description", "metaphor", "parable",
                         "allegory", "character sketch", "scene", "monologue"]

        themes = ["time travel", "artificial consciousness", "cultural exchange", "environmental harmony",
                 "technological singularity", "social justice", "personal growth", "human connection"]

        concepts = ["machine learning", "quantum physics", "democracy", "evolution", "creativity",
                   "consciousness", "sustainability", "globalization", "innovation", "empathy"]

        characters = ["a time traveler", "an AI", "a child", "an elderly person", "an alien", "a philosopher",
                     "a scientist", "an artist", "a teacher", "a storyteller"]

        formats = ["tutorial", "guide", "presentation", "workshop", "game", "simulation", "experiment",
                  "interactive experience", "learning journey", "collaborative project"]

        principles = ["collaboration", "sustainability", "innovation", "empathy", "resilience", "diversity",
                     "critical thinking", "creativity", "problem-solving", "communication"]

        scenarios = ["a world without internet", "first contact with aliens", "everyone has superpowers",
                    "time moves backwards", "emotions are visible", "memories can be shared"]

        aspects = ["the social dynamics", "the daily routines", "the technological adaptations",
                  "the cultural changes", "the emotional impact", "the ethical dilemmas"]

        systems = ["educational program", "social network", "transportation system", "communication tool",
                  "entertainment platform", "collaboration space", "learning environment"]

        purposes = ["connecting cultures", "solving climate change", "enhancing creativity", "building empathy",
                   "promoting understanding", "fostering innovation", "supporting wellbeing"]

        settings = ["a post-apocalyptic world", "a utopian society", "a space station", "a virtual reality",
                   "a time before technology", "a world of magic", "an underwater city"]

        elements = ["technology", "nature", "art", "science", "philosophy", "culture", "history", "future"]

        items = ["learning", "creativity", "communication", "problem-solving", "collaboration", "innovation",
                "adaptation", "resilience", "growth", "connection"]

        experiences = ["learning a new language", "traveling to space", "meeting your future self",
                      "experiencing synesthesia", "living in different time period", "having telepathy"]

        viewpoints = ["a child", "an AI", "an alien", "a historian from the future", "a philosopher",
                     "an artist", "a scientist", "someone from a different culture"]

        projects = ["a community garden", "a time capsule", "a cultural exchange", "an innovation lab",
                   "a storytelling platform", "a collaboration space", "a learning adventure"]

        future_techs = ["brain-computer interfaces", "quantum computing", "artificial general intelligence",
                       "nanotechnology", "genetic engineering", "fusion energy", "space elevators"]

        domains = ["education", "healthcare", "entertainment", "work", "relationships", "creativity",
                  "problem-solving", "communication", "transportation", "governance"]

        prompts = []
        for i in range(count):
            template = random.choice(templates)
            prompt = template.format(
                creative_type=random.choice(creative_types),
                theme=random.choice(themes),
                concept=random.choice(concepts),
                character=random.choice(characters),
                format=random.choice(formats),
                principle=random.choice(principles),
                scenario=random.choice(scenarios),
                aspect=random.choice(aspects),
                system=random.choice(systems),
                purpose=random.choice(purposes),
                setting=random.choice(settings),
                element1=random.choice(elements),
                element2=random.choice(elements),
                item1=random.choice(items),
                item2=random.choice(items),
                experience=random.choice(experiences),
                viewpoint=random.choice(viewpoints),
                project=random.choice(projects),
                future_tech=random.choice(future_techs),
                domain=random.choice(domains)
            )

            prompts.append({
                "id": f"creative_{i+1:03d}",
                "category": "creative",
                "prompt": prompt,
                "expected_concepts": self._extract_concepts(prompt, "creative")
            })

        return prompts

    def _extract_concepts(self, prompt: str, category: str) -> List[str]:
        """Extract expected concepts from prompt"""
        concept_maps = {
            "business": ["work", "company", "employee", "policy", "procedure", "office", "meeting"],
            "technical": ["system", "data", "software", "programming", "technology", "security", "performance"],
            "general": ["knowledge", "information", "learning", "understanding", "education", "science"],
            "complex": ["analysis", "strategy", "impact", "solution", "system", "factor", "outcome"],
            "creative": ["creativity", "imagination", "story", "art", "design", "innovation", "expression"]
        }

        base_concepts = concept_maps.get(category, [])
        prompt_lower = prompt.lower()

        found_concepts = []
        for concept in base_concepts:
            if concept in prompt_lower:
                found_concepts.append(concept)

        # Add category as primary concept
        found_concepts.insert(0, category)

        return found_concepts[:5]  # Limit to 5 concepts

    def generate_complete_dataset(self) -> List[Dict]:
        """Generate complete 1000+ test dataset"""
        print("🔄 Generating comprehensive 1000+ test dataset...")

        all_prompts = []
        all_prompts.extend(self.generate_business_prompts(200))
        all_prompts.extend(self.generate_technical_prompts(200))
        all_prompts.extend(self.generate_general_prompts(200))
        all_prompts.extend(self.generate_complex_prompts(200))
        all_prompts.extend(self.generate_creative_prompts(200))

        # Shuffle for random distribution
        random.shuffle(all_prompts)

        print(f"✅ Generated {len(all_prompts)} test prompts")
        print(f"📊 Categories: Business(200), Technical(200), General(200), Complex(200), Creative(200)")

        return all_prompts

    def save_dataset(self, prompts: List[Dict], filename: str):
        """Save dataset to file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "total_prompts": len(prompts),
                    "categories": {
                        "business": len([p for p in prompts if p["category"] == "business"]),
                        "technical": len([p for p in prompts if p["category"] == "technical"]),
                        "general": len([p for p in prompts if p["category"] == "general"]),
                        "complex": len([p for p in prompts if p["category"] == "complex"]),
                        "creative": len([p for p in prompts if p["category"] == "creative"])
                    },
                    "generated_at": "2025-09-20"
                },
                "prompts": prompts
            }, f, indent=2, ensure_ascii=False)

        print(f"💾 Dataset saved: {filename}")

def main():
    """Generate and save massive test dataset"""
    dataset = MassiveTestDataset()
    prompts = dataset.generate_complete_dataset()
    dataset.save_dataset(prompts, "results2/massive_test_dataset_1000.json")

    return len(prompts)

if __name__ == "__main__":
    total = main()
    print(f"\n🎯 Total prompts generated: {total}")
    print("🚀 Ready for massive ICAR testing!")