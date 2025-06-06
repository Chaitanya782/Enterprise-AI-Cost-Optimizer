import re


def _extract_financial_metrics_fixed(query: str, contextual_defaults: dict = None) -> dict:
    """Fixed version of financial data extraction"""
    metrics = {}
    query_lower = query.lower()
    multipliers = {'k': 1000, 'm': 1000000, 'b': 1000000000}

    # FIXED PATTERNS - More precise regex
    FINANCIAL_PATTERNS = [
        r'spending\s*\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*([kmb])?\s*(?:per\s+)?(?:month|monthly|mo)',
        r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:dollars?|usd)\s*([kmb])?\s*(?:per\s+)?(?:month|monthly|mo)?',
        r'budget\s*(?:of|is)?\s*\$?(\d+(?:,\d{3})*(?:\.\d+)?)([kmb])?\s*(?:per\s+)?(?:month|monthly|mo)?'
    ]

    PERCENT_PATTERNS = [
        r'(?:reduce|save|cut|decrease)\s*(?:by|up\s*to)?\s*(\d+(?:\.\d+)?)\s*%',
        r'(\d+(?:\.\d+)?)\s*%\s*(?:reduction|savings|decrease)',
        r'target\s*(?:of|is)?\s*(\d+(?:\.\d+)?)\s*%'
    ]

    print(f"Processing query: '{query_lower}'")

    # Extract from current query
    for i, pattern in enumerate(FINANCIAL_PATTERNS):
        print(f"Testing pattern {i + 1}: {pattern}")

        for match in re.finditer(pattern, query_lower, re.IGNORECASE):
            print(f"  Match found: '{match.group(0)}'")
            print(f"  Groups: {match.groups()}")

            # Parse the amount
            amount_str = match.group(1).replace(',', '')
            amount = float(amount_str)
            print(f"  Base amount: {amount}")

            # Check for multiplier - THIS IS THE CRITICAL FIX
            multiplier_char = match.group(2)
            print(f"  Multiplier group: '{multiplier_char}' (None: {multiplier_char is None})")

            if multiplier_char is not None and multiplier_char.strip():  # FIXED: Check for None AND empty string
                multiplier = multipliers.get(multiplier_char.lower().strip(), 1)
                amount *= multiplier
                print(f"  Applied multiplier {multiplier}: {amount}")
            else:
                print(f"  No multiplier applied")

            # Determine context
            context = match.group(0).lower()
            print(f"  Context: '{context}'")

            if any(term in context for term in ['month', 'monthly', 'mo']):
                metrics.update({'monthly_spend': amount, 'annual_spend': amount * 12})
                print(f"  Set as monthly spend: ${amount:,.2f}")
            elif any(term in context for term in ['year', 'annual', 'yearly']):
                metrics.update({'annual_spend': amount, 'monthly_spend': amount / 12})
                print(f"  Set as annual spend: ${amount:,.2f}")
            else:
                metrics['budget'] = amount
                print(f"  Set as budget: ${amount:,.2f}")

            break  # Take first match per pattern

    # Handle percentage patterns
    for pattern in PERCENT_PATTERNS:
        matches = re.findall(pattern, query_lower)
        if matches:
            metrics['target_reduction'] = float(matches[0]) / 100
            print(f"  Found target reduction: {metrics['target_reduction']:.1%}")
            break

    # Apply contextual defaults if no explicit values found
    if contextual_defaults and not metrics:
        for key in ['suggested_budget', 'suggested_monthly_spend', 'suggested_annual_spend']:
            if key in contextual_defaults:
                base_key = key.replace('suggested_', '')
                metrics[base_key] = contextual_defaults[key]
                print(f"  Applied contextual default {base_key}: {metrics[base_key]}")

    return metrics


# Test with your problematic query
test_query = "We're currently spending $8,000 monthly on manual processes"
result = _extract_financial_metrics_fixed(test_query)
print(f"\nFinal result: {result}")

# Test with other variations
test_cases = [
    "spending $8,000 monthly",
    "spending $8000 monthly",
    "spending $8k monthly",
    "spending $8,000k monthly",  # This should be 8 million
    "budget of $50,000",
    "We have 5000 dollars monthly budget"
]

print("\n=== TESTING MULTIPLE CASES ===")
for test in test_cases:
    print(f"\nTesting: '{test}'")
    result = _extract_financial_metrics_fixed(test)
    print(f"Result: {result}")