import asyncio
import dspy
from typing import Dict, List, Literal
from functools import partial


class EmpathyEvaluator(dspy.Signature):
    """Evaluates the empathy level of a doctor's response."""

    patient_question = dspy.InputField(desc="Întrebarea pacientului")
    doctor_response = dspy.InputField(desc="Răspunsul doctorului, care va fi evaluat")

    empathy: Literal["1", "2", "3", "4"] = dspy.OutputField(
        desc="""
        Empatie (1-4):
        1 - Răspuns urât, ceartă utilizatorul, îi face observații
        2 - Răspuns abrupt, direct, fără a menaja starea emoțională a utilizatorului, fără explicații
        3 - Răspuns politicos dar nu ia în considerare starea emoțională a utilizatorului, relativ scurt, cu puține explicații
        4 - Răspuns empatic, ține cont de starea emoțională a utilizatorului, explicativ, arată bunăvoință față de utilizator, încearcă să îl liniștească
        """
    )


class GrammaticalErrorsEvaluator(dspy.Signature):
    """Evaluates grammatical correctness of a doctor's response."""

    patient_question = dspy.InputField(desc="Întrebarea pacientului")
    doctor_response = dspy.InputField(desc="Răspunsul doctorului, care va fi evaluat")

    grammatical_errors: bool = dspy.OutputField(
        desc="Răspunsul conține erori gramaticale (true/false)"
    )


class AbbreviationsEvaluator(dspy.Signature):
    """Evaluates the use of abbreviations in a doctor's response."""

    patient_question = dspy.InputField(desc="Întrebarea pacientului")
    doctor_response = dspy.InputField(desc="Răspunsul doctorului, care va fi evaluat")

    abbreviations: bool = dspy.OutputField(
        desc="Răspunsul conține prescurtări (true/false)"
    )


class PunctuationErrorsEvaluator(dspy.Signature):
    """Evaluates punctuation correctness of a doctor's response."""

    patient_question = dspy.InputField(desc="Întrebarea pacientului")
    doctor_response = dspy.InputField(desc="Răspunsul doctorului, care va fi evaluat")

    punctuation_errors: bool = dspy.OutputField(
        desc="Răspunsul conține erori de punctuație (true/false)"
    )


class ProblemAddressingEvaluator(dspy.Signature):
    """Evaluates how well a doctor addressed patient's problems."""

    patient_question = dspy.InputField(desc="Întrebarea pacientului")
    doctor_response = dspy.InputField(desc="Răspunsul doctorului, care va fi evaluat")

    problems_addressed: Literal["1", "2", "3", "4", "5"] = dspy.OutputField(
        desc="""
    Toate Problemele (1-5):
    1 - Doctorul nu a adresat nici una din problemele pacientului, exemple includ răspunsuri precum "mergeți la doctor"
    2 - Doctorul a adresat o problemă principală, ignorând celelalte întrebări
    3 - Doctorul a adresat punctual majoritatea (aproximativ 80%) problemelor
    4 - Doctorul a adresat punctual toate problemele pacientului, fără alte completări
    5 - Doctorul a adresat toate problemele pacientului, inclusiv alte necunoscute, acoperind tot actul medical (cauze, tratament, rețetă, analize, pași următori)
    """
    )


class ClarificationEvaluator(dspy.Signature):
    """Evaluates the doctor's approach to getting more information."""

    patient_question = dspy.InputField(desc="Întrebarea pacientului")
    doctor_response = dspy.InputField(desc="Răspunsul doctorului, care va fi evaluat")

    clarifications: bool = dspy.OutputField(
        desc="""
    Clarificări (true/false): medicul a adresat doar întrebări de clarificare pacientului
    """
    )


class QuestionInsideResponseEvaluator(dspy.Signature):
    """Checks if the doctor asked question"""

    patient_question = dspy.InputField(desc="Întrebarea pacientului")
    doctor_response = dspy.InputField(desc="Răspunsul doctorului, care va fi evaluat")

    inside_questions: bool = dspy.OutputField(
        desc="""
    Întrebări în răspuns (true/false): medicul a răspuns la întrebarea pacientului, dar a adresat întrebări suplimentare în răspuns
    """
    )


class TreatmentShouldOfferEvaluator(dspy.Signature):
    """Checks if the patient should receive treatment"""

    patient_question = dspy.InputField(desc="Întrebarea pacientului")
    doctor_response = dspy.InputField(desc="Răspunsul doctorului, care va fi evaluat")

    treatment_should_offer: bool = dspy.OutputField(
        desc="Ar trebui medicul să ofere tratament în acest caz? (true/false)"
    )


class TreatmentOfferedEvaluator(dspy.Signature):
    """Checks if the response contains a recommended treatment"""

    patient_question = dspy.InputField(desc="Întrebarea pacientului")
    doctor_response = dspy.InputField(desc="Răspunsul doctorului, care va fi evaluat")

    treatment_did_offer: bool = dspy.OutputField(
        desc="A oferit medicul tratament? (true/false)"
    )


class PrescriptionEvaluator(dspy.Signature):
    """Evaluates prescription-related aspects of a doctor's response."""

    patient_question = dspy.InputField(desc="Întrebarea pacientului")
    doctor_response = dspy.InputField(desc="Răspunsul doctorului, care va fi evaluat")

    prescription_should_offer: bool = dspy.OutputField(
        desc="Pentru tratamentul oferit, ar trebui medicul să ofere rețetă? (true/false)"
    )


class CausesExplanationEvaluator(dspy.Signature):
    """Evaluates if the doctor explains causes of the medical condition."""

    patient_question = dspy.InputField(desc="Întrebarea pacientului")
    doctor_response = dspy.InputField(desc="Răspunsul doctorului, care va fi evaluat")

    explanation_causes: bool = dspy.OutputField(
        desc="Răspunsul menționează cauzele problemei? (true/false)"
    )


class SymptomsExplanationEvaluator(dspy.Signature):
    """Evaluates if the doctor explains symptoms of the condition."""

    patient_question = dspy.InputField(desc="Întrebarea pacientului")
    doctor_response = dspy.InputField(desc="Răspunsul doctorului, care va fi evaluat")

    explanation_symptoms: bool = dspy.OutputField(
        desc="Răspunsul menționează simptomele? (true/false)"
    )


class TreatmentExplanationEvaluator(dspy.Signature):
    """Evaluates if the doctor explains treatment options."""

    patient_question = dspy.InputField(desc="Întrebarea pacientului")
    doctor_response = dspy.InputField(desc="Răspunsul doctorului, care va fi evaluat")

    explanation_treatment: bool = dspy.OutputField(
        desc="Răspunsul menționează analize sau tratament? (true/false)"
    )


class RiskFactorsExplanationEvaluator(dspy.Signature):
    """Evaluates if the doctor explains risk factors."""

    patient_question = dspy.InputField(desc="Întrebarea pacientului")
    doctor_response = dspy.InputField(desc="Răspunsul doctorului, care va fi evaluat")

    explanation_risk_factors: bool = dspy.OutputField(
        desc="Răspunsul menționează factori de risc? (true/false)"
    )


class NextStepsExplanationEvaluator(dspy.Signature):
    """Evaluates if the doctor explains next steps."""

    patient_question = dspy.InputField(desc="Întrebarea pacientului")
    doctor_response = dspy.InputField(desc="Răspunsul doctorului, care va fi evaluat")

    explanation_next_steps: bool = dspy.OutputField(
        desc="Răspunsul explică pașii următori (inclusiv recomandări de unde să facă analize, etc)? (true/false)"
    )


class ChatGPTDetectorEvaluator(dspy.Signature):
    """Detects if the response was generated with ChatGPT."""

    patient_question = dspy.InputField(desc="Întrebarea pacientului")
    doctor_response = dspy.InputField(desc="Răspunsul doctorului, care va fi evaluat")

    generated_with_chatgpt: bool = dspy.OutputField(
        desc="Generat cu ChatGPT (true/false)"
    )


class SpecialtyReferralEvaluator(dspy.Signature):
    """Detects if doctor refers to another specialty."""

    patient_question = dspy.InputField(desc="Întrebarea pacientului")
    doctor_response = dspy.InputField(desc="Răspunsul doctorului, care va fi evaluat")

    other_specialty: bool = dspy.OutputField(
        desc="Medicul menționează în răspuns că nu poate ajuta pacientul întrucât cazul se pretează la o altă specialitate medicală (true/false)"
    )


class VisitOnlyEvaluator(dspy.Signature):
    """Detects if doctor only recommends in-person visit."""

    patient_question = dspy.InputField(desc="Întrebarea pacientului")
    doctor_response = dspy.InputField(desc="Răspunsul doctorului, care va fi evaluat")

    only_recommends_visit: bool = dspy.OutputField(
        desc="Medicul doar recomandă consult fizic și nu oferă alte informații/explicații (true/false)"
    )


class OnlineHelpLimitationEvaluator(dspy.Signature):
    """Detects if doctor mentions online help limitations."""

    patient_question = dspy.InputField(desc="Întrebarea pacientului")
    doctor_response = dspy.InputField(desc="Răspunsul doctorului, care va fi evaluat")

    cannot_help_online: bool = dspy.OutputField(
        desc="Medicul menționează că nu poate ajuta utilizatorul cu informații în mediul online (true/false)"
    )


class DoctorResponseScorerModule(dspy.Module):
    """Module that orchestrates multiple specialized evaluators for comprehensive assessment."""

    def __init__(self, cfg):
        super().__init__()
        # Initialize each specialized evaluator
        predict_module = cfg.predict_module
        self.scorers = {}
        for field, evaluator in field_to_evaluator.items():
            self.scorers[field] = predict_module(evaluator)

    async def scorer_async_call(self, scorer, patient_question, doctor_response):
        try:
            result = await scorer.acall(
                patient_question=patient_question, doctor_response=doctor_response
            )
            return result
        except Exception as e:
            print(e)
            return None

    async def aforward(
        self,
        patient_question: str,
        doctor_response: str,
        fields_to_score: Literal["all"] | List[str] = "all",
    ):

        tasks = []
        fields = []
        if fields_to_score == "all":
            for field, scorer in self.scorers.items():
                tasks.append(
                    self.scorer_async_call(scorer, patient_question, doctor_response)
                )
                fields.append(field)
        else:
            for f in fields_to_score:
                tasks.append(
                    self.scorer_async_call(
                        self.scorers[f],
                        patient_question,
                        doctor_response,
                    )
                )
                fields.append(f)

        predictions = await asyncio.gather(*tasks)

        final_result = dspy.Example(
            **{
                field: getattr(prediction, field) if prediction is not None else None
                for field, prediction in zip(fields, predictions)
            }
        )
        return final_result

    def forward(
        self,
        patient_question: str,
        doctor_response: str,
        fields: Literal["all"] | List[str] = "all",
    ):
        result = {}

        if fields == "all":
            for field, scorer in self.scorers.items():
                value = getattr(
                    scorer(
                        patient_question=patient_question,
                        doctor_response=doctor_response,
                    ),
                    field,
                )
                result[field] = value
        else:
            for f in fields:
                value = getattr(
                    self.scorers[f](
                        patient_question=patient_question,
                        doctor_response=doctor_response,
                    ),
                    f,
                )
                result[f] = value

        return dspy.Example(**result)


def numeric_metric(
    gold: Dict, prediction: Dict, trace=None, field_name: str = None, tolerance: int = 0
) -> float:
    """Generic metric for numeric fields with optional tolerance."""
    gold_value = gold.get(field_name)
    pred_value = prediction.get(field_name)

    if gold_value is not None and pred_value is not None:
        return float(abs(int(gold_value) - int(pred_value)) <= tolerance)
    return 0.0


def boolean_metric(
    gold: Dict, prediction: Dict, trace=None, field_name: str = None
) -> float:
    """Generic metric for boolean fields."""
    gold_value = gold.get(field_name)
    pred_value = prediction.get(field_name)

    if gold_value is not None and pred_value is not None:
        return float(gold_value == pred_value)
    return 0.0


# Module-specific metrics map for use during evaluation
metric_map = {
    "empathy": partial(numeric_metric, field_name="empathy"),
    "problems_addressed": partial(numeric_metric, field_name="problems_addressed"),
    "grammatical_errors": partial(boolean_metric, field_name="grammatical_errors"),
    "abbreviations": partial(boolean_metric, field_name="abbreviations"),
    "punctuation_errors": partial(boolean_metric, field_name="punctuation_errors"),
    "clarifications": partial(boolean_metric, field_name="clarifications"),
    "inside_questions": partial(boolean_metric, field_name="inside_questions"),
    "treatment_should_offer": partial(
        boolean_metric, field_name="treatment_should_offer"
    ),
    "treatment_did_offer": partial(boolean_metric, field_name="treatment_did_offer"),
    "prescription_should_offer": partial(
        boolean_metric, field_name="prescription_should_offer"
    ),
    "explanation_causes": partial(boolean_metric, field_name="explanation_causes"),
    "explanation_symptoms": partial(boolean_metric, field_name="explanation_symptoms"),
    "explanation_treatment": partial(
        boolean_metric, field_name="explanation_treatment"
    ),
    "explanation_risk_factors": partial(
        boolean_metric, field_name="explanation_risk_factors"
    ),
    "explanation_next_steps": partial(
        boolean_metric, field_name="explanation_next_steps"
    ),
    "generated_with_chatgpt": partial(
        boolean_metric, field_name="generated_with_chatgpt"
    ),
    "other_specialty": partial(boolean_metric, field_name="other_specialty"),
    "only_recommends_visit": partial(
        boolean_metric, field_name="only_recommends_visit"
    ),
    "cannot_help_online": partial(boolean_metric, field_name="cannot_help_online"),
}

description_map = {
    # Basic quality metrics
    "grammatical_errors": GrammaticalErrorsEvaluator.fields[
        "grammatical_errors"
    ].json_schema_extra["desc"],
    "abbreviations": AbbreviationsEvaluator.fields["abbreviations"].json_schema_extra[
        "desc"
    ],
    "punctuation_errors": PunctuationErrorsEvaluator.fields[
        "punctuation_errors"
    ].json_schema_extra["desc"],
    "empathy": EmpathyEvaluator.fields["empathy"].json_schema_extra["desc"],
    "problems_addressed": ProblemAddressingEvaluator.fields[
        "problems_addressed"
    ].json_schema_extra["desc"],
    "clarifications": ClarificationEvaluator.fields["clarifications"].json_schema_extra[
        "desc"
    ],
    "inside_questions": QuestionInsideResponseEvaluator.fields[
        "inside_questions"
    ].json_schema_extra["desc"],
    # Treatment metrics
    "treatment_should_offer": TreatmentShouldOfferEvaluator.fields[
        "treatment_should_offer"
    ].json_schema_extra["desc"],
    "treatment_did_offer": TreatmentOfferedEvaluator.fields[
        "treatment_did_offer"
    ].json_schema_extra["desc"],
    "prescription_should_offer": PrescriptionEvaluator.fields[
        "prescription_should_offer"
    ].json_schema_extra["desc"],
    # Explanation metrics
    "explanation_causes": CausesExplanationEvaluator.fields[
        "explanation_causes"
    ].json_schema_extra["desc"],
    "explanation_symptoms": SymptomsExplanationEvaluator.fields[
        "explanation_symptoms"
    ].json_schema_extra["desc"],
    "explanation_treatment": TreatmentExplanationEvaluator.fields[
        "explanation_treatment"
    ].json_schema_extra["desc"],
    "explanation_risk_factors": RiskFactorsExplanationEvaluator.fields[
        "explanation_risk_factors"
    ].json_schema_extra["desc"],
    "explanation_next_steps": NextStepsExplanationEvaluator.fields[
        "explanation_next_steps"
    ].json_schema_extra["desc"],
    "generated_with_chatgpt": ChatGPTDetectorEvaluator.fields[
        "generated_with_chatgpt"
    ].json_schema_extra["desc"],
    "other_specialty": SpecialtyReferralEvaluator.fields[
        "other_specialty"
    ].json_schema_extra["desc"],
    "only_recommends_visit": VisitOnlyEvaluator.fields[
        "only_recommends_visit"
    ].json_schema_extra["desc"],
    "cannot_help_online": OnlineHelpLimitationEvaluator.fields[
        "cannot_help_online"
    ].json_schema_extra["desc"],
}


type_map: Dict[str, Literal["int", "bool", "list"]] = {
    "empathy": "int",
    "problems_addressed": "int",
    "grammatical_errors": "bool",
    "abbreviations": "bool",
    "punctuation_errors": "bool",
    "medical_terms": "list",
    "clarifications": "bool",
    "inside_questions": "bool",
    "treatment_should_offer": "bool",
    "treatment_did_offer": "bool",
    "prescription_should_offer": "bool",
    "explanation_causes": "bool",
    "explanation_symptoms": "bool",
    "explanation_treatment": "bool",
    "explanation_risk_factors": "bool",
    "explanation_next_steps": "bool",
    "generated_with_chatgpt": "bool",
    "other_specialty": "bool",
    "only_recommends_visit": "bool",
    "cannot_help_online": "bool",
}

evaluators = [
    EmpathyEvaluator,
    ProblemAddressingEvaluator,
    GrammaticalErrorsEvaluator,
    AbbreviationsEvaluator,
    PunctuationErrorsEvaluator,
    ClarificationEvaluator,
    QuestionInsideResponseEvaluator,
    TreatmentShouldOfferEvaluator,
    PrescriptionEvaluator,
    CausesExplanationEvaluator,
    SymptomsExplanationEvaluator,
    TreatmentExplanationEvaluator,
    RiskFactorsExplanationEvaluator,
    NextStepsExplanationEvaluator,
    ChatGPTDetectorEvaluator,
    SpecialtyReferralEvaluator,
    VisitOnlyEvaluator,
    OnlineHelpLimitationEvaluator,
]

field_to_evaluator = {
    "empathy": EmpathyEvaluator,
    "problems_addressed": ProblemAddressingEvaluator,
    "grammatical_errors": GrammaticalErrorsEvaluator,
    "abbreviations": AbbreviationsEvaluator,
    "punctuation_errors": PunctuationErrorsEvaluator,
    "clarifications": ClarificationEvaluator,
    "inside_questions": QuestionInsideResponseEvaluator,
    "treatment_should_offer": TreatmentShouldOfferEvaluator,
    "treatment_did_offer": TreatmentOfferedEvaluator,
    "prescription_should_offer": PrescriptionEvaluator,
    "explanation_causes": CausesExplanationEvaluator,
    "explanation_symptoms": SymptomsExplanationEvaluator,
    "explanation_treatment": TreatmentExplanationEvaluator,
    "explanation_risk_factors": RiskFactorsExplanationEvaluator,
    "explanation_next_steps": NextStepsExplanationEvaluator,
    "generated_with_chatgpt": ChatGPTDetectorEvaluator,
    "other_specialty": SpecialtyReferralEvaluator,
    "only_recommends_visit": VisitOnlyEvaluator,
    "cannot_help_online": OnlineHelpLimitationEvaluator,
}


def check_for_needed_recommendation(field, row):
    if field == "empathy":
        if row[field] is not None:
            return int(row[field]) < 4

    if field == "problems_addressed":
        if row[field] is not None:
            return int(row[field]) < 5

    if field in ["grammatical_errors", "abbreviations", "punctuation_errors"]:
        return row[field] == True

    if (
        field == "treatment_did_offer"
        and row[field] == False
        and row["treatment_should_offer"] == True
    ):
        return True

    if field == "prescription_should_offer" and row[field] == True:
        return True

    if field in [
        "explanation_causes",
        "explanation_symptoms",
        "explanation_risk_factors",
        "explanation_next_steps",
    ]:
        return row[field] == False

    if field in [
        "clarifications",
        "inside_questions",
        "generated_with_chatgpt",
        "other_specialty",
        "only_recommends_visit",
        "cannot_help_online",
    ]:
        return row[field] == True

    return False
