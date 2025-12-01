"""
AI-powered investment analysis using OpenAI.
"""

import os
from openai import OpenAI
from typing import Dict, Any


class PropertyInvestmentAnalyzer:
    """
    Analyzes property data and generates AI-powered investment recommendations.
    """

    def __init__(self):
        """
        Initialize the analyzer with OpenAI API key.

        Args:
            api_key: OpenAI API key. If not provided, will look for OPENAI_API_KEY env variable.
        """
        self.api_key = os.getenv("OPENAI_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.client = OpenAI(api_key=self.api_key)

    def generate_investment_summary(
        self,
        address: str,
        price_pred: float = None,
        occ_pred: float = None,
        rev_pred: float = None,
        property_features: Dict[str, Any] = None,
        model: str = "gpt-4o"
    ) -> str:
        """
        Generate an AI-powered investment summary for a Chicago STR property.

        Args:
            address: Property address
            price_pred: ML-predicted nightly price
            occ_pred: ML-predicted annual occupancy (days)
            rev_pred: ML-predicted annual revenue
            property_features: Dictionary of additional property features
            model: OpenAI model to use (default: gpt-4o-mini for cost efficiency)

        Returns:
            Investment summary as a string
        """
        # Build the prompt
        prompt = self._build_analysis_prompt(
            address=address,
            price_pred=price_pred,
            occ_pred=occ_pred,
            rev_pred=rev_pred,
            property_features=property_features
        )

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a real estate investment advisor specializing in "
                            "short-term rental (STR) properties in Chicago. Provide clear, actionable "
                            "investment recommendations based on property data and ML predictions. "
                            "Be concise but thorough, focusing on key investment metrics and risks. "
                            "Format your response as plain text with clear paragraph breaks. "
                            "Do not use markdown formatting like **bold** or bullet points."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=500
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Error generating AI summary: {str(e)}"

    def _build_analysis_prompt(
        self,
        address: str,
        price_pred: float = None,
        occ_pred: float = None,
        rev_pred: float = None,
        property_features: Dict[str, Any] = None
    ) -> str:
        """
        Build a detailed prompt for the AI model.
        """

        # Build ML predictions section
        ml_predictions = []
        if price_pred:
            ml_predictions.append(
                f"- Predicted Nightly Rate: ${price_pred:.2f}")
        if occ_pred:
            occupancy_rate = (occ_pred / 365) * 100
            ml_predictions.append(
                f"- Predicted Annual Occupancy: {occ_pred:.0f} days ({occupancy_rate:.1f}%)")
        if rev_pred:
            ml_predictions.append(
                f"- Predicted Annual Revenue: ${rev_pred:,.2f}")

        predictions_text = "\n".join(
            ml_predictions) if ml_predictions else "ML predictions not available"

        # Build property features section
        features_text = "Limited property information available"
        if property_features:
            feature_lines = []

            # Location features
            if property_features.get('dist_to_airport_km'):
                feature_lines.append(
                    f"- Distance to Airport: {property_features['dist_to_airport_km']:.1f} km")
            if property_features.get('dist_to_train_km'):
                feature_lines.append(
                    f"- Distance to Train: {property_features['dist_to_train_km']:.1f} km")
            if property_features.get('dist_to_city_center_km'):
                feature_lines.append(
                    f"- Distance to City Center: {property_features['dist_to_city_center_km']:.1f} km")

            # Demographics
            if property_features.get('median_income'):
                feature_lines.append(
                    f"- Area Median Income: ${property_features['median_income']:,.0f}")
            if property_features.get('median_home_value'):
                feature_lines.append(
                    f"- Area Median Home Value: ${property_features['median_home_value']:,.0f}")

            if feature_lines:
                features_text = "\n".join(feature_lines)

        prompt = f"""
Analyze this Chicago short-term rental property for investment potential:

PROPERTY INFORMATION
Address: {address}

ML MODEL PREDICTIONS
{predictions_text}

PROPERTY & AREA FEATURES
{features_text}

TASK
Provide a concise investment recommendation (250-350 words) covering:

1. Investment Rating: Rate as "Strong Buy", "Buy", "Hold", "Caution", or "Avoid"

2. Key Strengths: List 2-3 main positive factors for STR investment

3. Key Concerns: List 2-3 main risk factors or concerns

4. Performance Analysis: Comment on whether the ML predictions seem reasonable given the location and features

5. Bottom Line: Final recommendation in 1-2 sentences

Focus on actionable insights specific to Chicago's short-term rental market.
Consider the ML predictions in context of the property's location and neighborhood characteristics.
"""

        return prompt.strip()
