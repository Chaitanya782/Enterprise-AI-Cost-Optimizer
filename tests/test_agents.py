"""
Test script for agent functionality with new Lyzr API
"""
import sys
from pathlib import Path
import uuid
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.lyzr_client import get_lyzr_client
from app.config import config
from utils.logger import logger


def test_lyzr_agent():
    """Test Lyzr agent chat functionality"""
    print("🧪 Testing Lyzr Agent System...\n")

    # Validate configuration first
    if not config.validate():
        print("❌ Configuration validation failed. Please check your .env file.")
        return False

    try:
        # Initialize the client
        client = get_lyzr_client()
        print(f"✅ Lyzr client initialized successfully")
        print(f"🤖 Agent ID: {config.lyzr_agent_id}")
        print(f"👤 User ID: {config.lyzr_user_id}\n")

        # Generate a unique session ID for this test
        session_id = f"test-session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Test 1: Task Analysis
        print("1️⃣ Testing Task Analysis...")
        try:
            task_prompt = """
            Analyze this business workflow for AI automation opportunities:
            
            Our customer service team handles 500 support tickets daily.
            Each agent spends 15 minutes per ticket on average.
            Common tasks include: answering FAQs, order status checks, 
            refund processing, and technical troubleshooting.
            
            Please identify:
            1. Tasks suitable for AI automation
            2. Potential time savings
            3. Implementation complexity
            4. Expected ROI
            """

            result = client.chat(
                message=task_prompt,
                session_id=f"{session_id}-task-analysis"
            )

            print("✅ Task Analysis Response received:")
            response_text = result.get('response', result.get('message', str(result)))
            print(f"{response_text[:500]}...\n")

        except Exception as e:
            print(f"❌ Task Analysis Error: {str(e)}\n")

        # Test 2: Cost Calculation
        print("2️⃣ Testing Cost Calculation...")
        try:
            cost_prompt = """
            Calculate the costs for implementing an AI customer service chatbot with these specifications:
            
            - Use case: Customer Service Chatbot
            - Expected daily requests: 1,000
            - Average input tokens per request: 150
            - Average output tokens per request: 200
            - Operating days per year: 365
            
            Please provide:
            1. Cost breakdown by different LLM providers (GPT-4, Claude, Gemini)
            2. Monthly and annual cost estimates
            3. Cost optimization recommendations
            4. Comparison with human agent costs
            """

            result = client.chat(
                message=cost_prompt,
                session_id=f"{session_id}-cost-calc"
            )

            print("✅ Cost Calculation Response received:")
            response_text = result.get('response', result.get('message', str(result)))
            print(f"{response_text[:500]}...\n")

        except Exception as e:
            print(f"❌ Cost Calculation Error: {str(e)}\n")

        # Test 3: ROI Estimation
        print("3️⃣ Testing ROI Estimation...")
        try:
            roi_prompt = """
            Calculate ROI for this AI implementation project:
            
            Project: AI Customer Service Implementation
            Implementation Cost: $50,000
            Annual Benefits:
            - Labor Cost Savings: $120,000
            - Increased Sales (24/7 availability): $30,000
            - Reduced Error Costs: $15,000
            Timeline: 3 years
            
            Please provide:
            1. ROI percentage and payback period
            2. Net Present Value (NPV) analysis
            3. Break-even analysis
            4. Risk factors and mitigation strategies
            5. Year-by-year financial projections
            """

            result = client.chat(
                message=roi_prompt,
                session_id=f"{session_id}-roi-analysis"
            )

            print("✅ ROI Estimation Response received:")
            response_text = result.get('response', result.get('message', str(result)))
            print(f"{response_text[:500]}...\n")

        except Exception as e:
            print(f"❌ ROI Estimation Error: {str(e)}\n")

        # Test 4: General AI Cost Optimization
        print("4️⃣ Testing General AI Cost Optimization...")
        try:
            optimization_prompt = """
            I'm running a SaaS company with 10,000 monthly active users. We're considering integrating AI features:
            
            - AI-powered search and recommendations
            - Automated content generation
            - Customer support chatbot
            - Data analysis and insights
            
            What's the most cost-effective approach to implement these features while maintaining quality?
            Please provide a prioritized implementation roadmap with cost estimates.
            """

            result = client.chat(
                message=optimization_prompt,
                session_id=f"{session_id}-optimization"
            )

            print("✅ AI Cost Optimization Response received:")
            response_text = result.get('response', result.get('message', str(result)))
            print(f"{response_text[:500]}...\n")

        except Exception as e:
            print(f"❌ AI Cost Optimization Error: {str(e)}\n")

        print("✅ All agent tests completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Critical Error: {str(e)}")
        logger.error(f"Test failed with error: {str(e)}")
        return False


def test_session_continuity():
    """Test session continuity with follow-up questions"""
    print("\n🔄 Testing Session Continuity...\n")

    try:
        client = get_lyzr_client()
        session_id = f"continuity-test-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # First message
        print("📝 Sending initial question...")
        result1 = client.chat(
            message="I want to implement an AI chatbot for my e-commerce store with 1000 daily visitors. What would be the estimated monthly cost?",
            session_id=session_id
        )

        response1 = result1.get('response', result1.get('message', str(result1)))
        print(f"Response 1: {response1[:300]}...\n")

        # Follow-up message
        print("📝 Sending follow-up question...")
        result2 = client.chat(
            message="What if I scale it to 5000 daily visitors? How would the costs change?",
            session_id=session_id
        )

        response2 = result2.get('response', result2.get('message', str(result2)))
        print(f"Response 2: {response2[:300]}...\n")

        print("✅ Session continuity test completed!")

    except Exception as e:
        print(f"❌ Session continuity test failed: {str(e)}")


def main():
    """Main test function"""
    print("🚀 Enterprise AI Cost Optimizer - Agent Testing\n")
    print("=" * 60)

    # Test basic agent functionality
    success = test_lyzr_agent()

    if success:
        # Test session continuity
        test_session_continuity()

        print("\n" + "=" * 60)
        print("🎉 All tests completed! The agent system is ready for use.")
        print("\n💡 Tips for using the system:")
        print("   • Use specific, detailed prompts for better results")
        print("   • Include numerical data when asking for cost calculations")
        print("   • Maintain session IDs for conversation continuity")
        print("   • Check your API usage to stay within rate limits")
    else:
        print("\n❌ Tests failed. Please check your configuration and try again.")


if __name__ == "__main__":
    main()
