"""Ready-made classification tasks matching the files in sample_data/."""

PRESETS = {
    "support_tickets": {
        "labels": ["billing", "technical", "account", "shipping", "other"],
        "instruction": "Classify the customer support ticket by the department that should handle it.",
        "output_column": "category",
    },
    "reviews": {
        "labels": ["positive", "negative", "neutral"],
        "instruction": "Classify the sentiment of the product review.",
        "output_column": "sentiment",
    },
    "transactions": {
        "labels": ["groceries", "dining", "transport", "utilities", "entertainment", "shopping", "other"],
        "instruction": "Classify the bank transaction into a spending category.",
        "output_column": "spend_category",
    },
}
