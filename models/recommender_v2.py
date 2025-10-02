import asyncio
import dspy
from typing import List, Literal
from models.prompt_score_v4 import (
    description_map,
    check_for_needed_recommendation,
)


class EmpathyRecommender(dspy.Signature):
    """Creează recomandări pentru îmbunătățirea nivelului de empatie din răspunsul medicului.
    Recomandarile trebuie sa inceapa cu "Pentru a crește nivelul de empatie al răspunsului, puteți avea în vedere următoarele aspecte: <lista, separate pe cate o linie, maxim 3>".
    Fiecare element din lista ar trebui sa inceapa cu "- Ati putea", "- Ar ajuta", "- Ar fi bine" etc.
    Recomandările trebuie să fie în limba română si sa fie succinte.
    Nu oferi recomandări medicale, nu ii spune doctorului ce sa faca, doar semnaleaza posibile probleme din raspunsul lui.
    """

    patient_question = dspy.InputField(desc="Întrebarea pacientului")
    doctor_response = dspy.InputField(desc="Răspunsul doctorului, care va fi evaluat")
    score = dspy.InputField(desc=description_map["empathy"])
    recommendation = dspy.OutputField(desc="Recomandare pentru îmbunătățirea empatiei")


class ProblemAddressingRecommender(dspy.Signature):
    """Identifică problemele neadresate din întrebarea pacientului, precizându-le într-o listă.
    Recomandările trebuie să fie în limba română și să nu înceapă cu "Pentru a îmbunătăți scorul...".
    Recomandarea poate începe cu "Răspunsul ar putea beneficia de următoarele detalii: <listă probleme neadresate, separate pe câte un rând, maxim 4>"
    """

    patient_question = dspy.InputField(desc="Întrebarea pacientului")
    doctor_response = dspy.InputField(desc="Răspunsul doctorului, care va fi evaluat")
    score = dspy.InputField(desc=description_map["problems_addressed"])
    recommendation = dspy.OutputField(
        desc="Recomandare pentru adresarea completă a problemelor"
    )


class GrammaticalErrorsRecommender(dspy.Signature):
    """Creează recomandări pentru corectarea erorilor gramaticale din răspunsul medicului.
    Raspunsul poate incepe cu: "Aveti cateva probleme pe care le-ati putea corecta: <lista probleme, separate pe cate 1 linie, maxim 3>"
    Concentrează-te strict pe corectitudinea gramaticală, nu pe conținutul medical. Semnaleaza posibilele probleme din raspuns sub forma unei liste
    """

    patient_question = dspy.InputField(desc="Întrebarea pacientului")
    doctor_response = dspy.InputField(desc="Răspunsul doctorului, care va fi evaluat")
    score = dspy.InputField(desc=description_map["grammatical_errors"])
    recommendation = dspy.OutputField(
        desc="Recomandare pentru corectarea erorilor gramaticale"
    )


class AbbreviationsRecommender(dspy.Signature):
    """Creează recomandări pentru evitarea abrevierilor în răspunsul medicului.

    Recomandările trebuie să:
    - Identifice toate abrevierile utilizate în răspuns
    - Ofere formele complete ale termenilor abreviați
    - Explice de ce abrevierile pot fi confuze pentru pacienți
    - Sugereze cum să comunice aceleași informații fără abrevieri

    Recomandarile trebuie formulate in 1-2 fraze.
    Semnaleaza maxim 4 cele mai importante probleme.
    Recomandările trebuie să fie în limba română.
    Concentrează-te pe îmbunătățirea clarității comunicării pentru pacienți. Semnaleaza posibilele probleme din raspuns sub forma unei liste
    """

    patient_question = dspy.InputField(desc="Întrebarea pacientului")
    doctor_response = dspy.InputField(desc="Răspunsul doctorului, care va fi evaluat")
    score = dspy.InputField(desc=description_map["abbreviations"])
    recommendation = dspy.OutputField(desc="Recomandare pentru evitarea abrevierilor")


class PunctuationErrorsRecommender(dspy.Signature):
    """Creează recomandări pentru corectarea erorilor de punctuație din răspunsul medicului.
    Recomandarile trebuie sa inceapa: "Aveti cateva probleme de punctuatie: <lista probleme, separate pe cate o linie, maxim 3>"
    Recomandările trebuie să fie în limba română.
    Concentrează-te strict pe punctuație, nu pe conținutul medical. Semnaleaza posibilele probleme din raspuns sub forma unei liste
    """

    patient_question = dspy.InputField(desc="Întrebarea pacientului")
    doctor_response = dspy.InputField(desc="Răspunsul doctorului, care va fi evaluat")
    score = dspy.InputField(desc=description_map["punctuation_errors"])
    recommendation = dspy.OutputField(
        desc="Recomandare pentru corectarea erorilor de punctuație"
    )


class ClarificationRecommender(dspy.Signature):
    """Creează o recomandare în cazul în care doctorul cere clarificări
    Recomandea trebuie să fie în limba română.

    Răspunsul trebuie să înceapă similar cu "Vă rugăm să folosiți funcția de solicitare clarificări de mai sus, pentru a trimite aceste întrebări pacientului: <listă de întrebări din răspuns, pe un rând separat, cu - la inceput>.
    """

    patient_question = dspy.InputField(desc="Întrebarea pacientului")
    doctor_response = dspy.InputField(desc="Răspunsul doctorului, care va fi evaluat")
    score = dspy.InputField(desc=description_map["clarifications"])
    recommendation = dspy.OutputField(
        desc="Recomandare pentru a echilibra clarificările cu informațiile oferite"
    )


class QuestionInsideResponseRecommender(dspy.Signature):
    """Creează recomandări pentru cazurile când medicul include întrebări în răspunsul său.
    Recomandea trebuie să fie în limba română.

    Răspunsul trebuie sa contina "Vă rugăm să folosiți funcția de solicitare clarificări de mai sus, pentru a trimite aceste întrebări pacientului: <listă de întrebări din răspuns, pe un rând separat, cu - la inceput>.

    Recomandăm:
    1. Utilizarea funcționalității de solicitare clarificări pentru adresarea întrebărilor;
    2. Furnizarea răspunsului ulterior clarificărilor primite."
    """

    patient_question = dspy.InputField(desc="Întrebarea pacientului")
    doctor_response = dspy.InputField(desc="Răspunsul doctorului, care va fi evaluat")
    score = dspy.InputField(desc=description_map["inside_questions"])
    recommendation = dspy.OutputField(
        desc="Recomandare pentru evitarea întrebărilor în răspuns"
    )


class TreatmentOfferedRecommender(dspy.Signature):
    """Creează recomandări pentru cazurile când medicul nu a oferit tratament deși ar fi trebuit.
    Identifica aspectele din întrebarea pacientului care necesitau sugestii de tratament.
    Recomandarile pot incepe cu text similar cu : "Răspunsul ar putea beneficia de includerea unei recomandări de tratament, acolo unde este posibil."
    Recomandările trebuie să fie în limba română.
    Nu oferi recomandări medicale specifice, doar îmbunătățiri ale modului de comunicare.
    """

    patient_question = dspy.InputField(desc="Întrebarea pacientului")
    doctor_response = dspy.InputField(desc="Răspunsul doctorului, care va fi evaluat")
    score = dspy.InputField(desc=description_map["treatment_did_offer"])
    recommendation = dspy.OutputField(
        desc="Recomandare pentru includerea tratamentului în răspuns"
    )


class PrescriptionRecommender(dspy.Signature):
    """Creează recomandări pentru cazurile când medicul ar trebui să menționeze necesitatea unei rețete.
    Recomandarea trebuie sa aiba forma: "În cazul în care medicamentul/medicamentele <lista medicamente> necesită rețetă, o puteți atașa."
    """

    patient_question = dspy.InputField(desc="Întrebarea pacientului")
    doctor_response = dspy.InputField(desc="Răspunsul doctorului, care va fi evaluat")
    score = dspy.InputField(desc=description_map["prescription_should_offer"])
    recommendation = dspy.OutputField(
        desc="Recomandare pentru menționarea necesității rețetei"
    )


class CausesExplanationRecommender(dspy.Signature):
    """
    Recomandarea trebuie sa aiba forma: "Răspunsul ar putea beneficia de o detaliere a posibilelor cauze ale <afectiunii>."
    Recomandarea trebuie să fie în limba română.
    Nu oferi recomandări medicale specifice.
    """

    patient_question = dspy.InputField(desc="Întrebarea pacientului")
    doctor_response = dspy.InputField(desc="Răspunsul doctorului, care va fi evaluat")
    score = dspy.InputField(desc=description_map["explanation_causes"])
    recommendation = dspy.OutputField(desc="Recomandare pentru explicarea cauzelor")


class SymptomsExplanationRecommender(dspy.Signature):
    """
    Recomandarea trebuie sa aiba forma : "Răspunsul ar putea beneficia de o detaliere a simptomelor frecvent asociate cu <afecțiunea>."
    Recomandările trebuie să fie în limba română.
    Nu oferi recomandări medicale specifice.
    """

    patient_question = dspy.InputField(desc="Întrebarea pacientului")
    doctor_response = dspy.InputField(desc="Răspunsul doctorului, care va fi evaluat")
    score = dspy.InputField(desc=description_map["explanation_symptoms"])
    recommendation = dspy.OutputField(desc="Recomandare pentru explicarea simptomelor")


class TreatmentExplanationRecommender(dspy.Signature):
    """Creează recomandări pentru îmbunătățirea explicațiilor despre tratament și analize.

    Recomandările trebuie să:
    - Sugereze cum să explice mai detaliat opțiunile de tratament
    - Propună modalități de a descrie scopul și beneficiile analizelor recomandate
    - Indice cum să prezinte așteptările realiste privind eficiența tratamentului
    - Exemplifice cum să explice procesul de tratament într-un mod accesibil

    Recomandarile trebuie formulate in 1-2 fraze.
    Recomandarile pot incepe cu text similar cu : "Ati putea sa..."
    Recomandările trebuie să fie în limba română și să nu înceapă cu "Pentru a îmbunătăți scorul...".
    Nu oferi recomandări medicale specifice, doar îmbunătățiri ale modului de comunicare.
    """

    patient_question = dspy.InputField(desc="Întrebarea pacientului")
    doctor_response = dspy.InputField(desc="Răspunsul doctorului, care va fi evaluat")
    score = dspy.InputField(desc=description_map["explanation_treatment"])
    recommendation = dspy.OutputField(
        desc="Recomandare pentru explicarea tratamentului și analizelor"
    )


class RiskFactorsExplanationRecommender(dspy.Signature):
    """Creează recomandări pentru îmbunătățirea explicațiilor despre factorii de risc.
    Recomandarile trebuie sa aiba forma: "Răspunsul ar putea beneficia de informații legate de factorii de risc."
    Nu oferi recomandări medicale specifice, doar îmbunătățiri ale modului de comunicare.
    """

    patient_question = dspy.InputField(desc="Întrebarea pacientului")
    doctor_response = dspy.InputField(desc="Răspunsul doctorului, care va fi evaluat")
    score = dspy.InputField(desc=description_map["explanation_risk_factors"])
    recommendation = dspy.OutputField(
        desc="Recomandare pentru explicarea factorilor de risc"
    )


class NextStepsExplanationRecommender(dspy.Signature):
    """Recomandarea poate avea forma: "Răspunsul ar putea beneficia de includerea unor informații privind pașii următori."
    Nu oferi recomandări medicale specifice, doar îmbunătățiri ale modului de comunicare.
    """

    patient_question = dspy.InputField(desc="Întrebarea pacientului")
    doctor_response = dspy.InputField(desc="Răspunsul doctorului, care va fi evaluat")
    score = dspy.InputField(desc=description_map["explanation_next_steps"])
    recommendation = dspy.OutputField(
        desc="Recomandare pentru explicarea pașilor următori"
    )


class ChatGPTDetectorRecommender(dspy.Signature):
    """Creează recomandări pentru cazurile când răspunsul pare generat cu ChatGPT.

    Recomandările trebuie să:
    - Identifice caracteristicile care fac răspunsul să pară generat automat
    - Sugereze cum să personalizeze limbajul pentru a părea mai autentic
    - Propună modificări pentru a adăuga un ton profesional dar uman
    - Indice cum să adapteze structura și stilul la contextul specific al conversației

    Recomandarile trebuie formulate in 1-2 fraze.
    Recomandările trebuie să fie în limba română și să nu înceapă cu "Pentru a îmbunătăți scorul...".
    Concentrează-te pe îmbunătățirea autenticității comunicării.
    """

    patient_question = dspy.InputField(desc="Întrebarea pacientului")
    doctor_response = dspy.InputField(desc="Răspunsul doctorului, care va fi evaluat")
    score = dspy.InputField(desc=description_map["generated_with_chatgpt"])
    recommendation = dspy.OutputField(
        desc="Recomandare pentru a face răspunsul să pară mai autentic"
    )


class SpecialtyReferralRecommender(dspy.Signature):
    """
    Recomandarea trebuie sa aiba forma: "În cazul în care considerați că întrebarea adresată ține de o altă specialitate medicală, vă rugăm să o refuzați, conform procedurii."
    """

    patient_question = dspy.InputField(desc="Întrebarea pacientului")
    doctor_response = dspy.InputField(desc="Răspunsul doctorului, care va fi evaluat")
    score = dspy.InputField(desc=description_map["other_specialty"])
    recommendation = dspy.OutputField(
        desc="Recomandare pentru îmbunătățirea referirii către altă specialitate"
    )


class VisitOnlyRecommender(dspy.Signature):
    """Recomandarea trebuie să fie în limba română și să înceapă cu "În cazul în care considerați că întrebarea adresată nu se pretează unui consult online, fiind necesară o consultație fizică, vă rugăm să o refuzați, conform procedurii."."""

    patient_question = dspy.InputField(desc="Întrebarea pacientului")
    doctor_response = dspy.InputField(desc="Răspunsul doctorului, care va fi evaluat")
    score = dspy.InputField(desc=description_map["only_recommends_visit"])
    recommendation = dspy.OutputField(
        desc="Recomandare pentru echilibrarea recomandării consultului cu informații utile"
    )


class OnlineHelpLimitationRecommender(dspy.Signature):
    """Recomandarea trebuie să fie în limba română și să înceapă cu "În cazul în care considerați că nu puteți ajuta în mediul online, vă rugăm să refuzați conversația, conform procedurii."."""

    patient_question = dspy.InputField(desc="Întrebarea pacientului")
    doctor_response = dspy.InputField(desc="Răspunsul doctorului, care va fi evaluat")
    score = dspy.InputField(desc=description_map["cannot_help_online"])
    recommendation = dspy.OutputField(
        desc="Recomandare pentru abordarea limitărilor ajutorului online"
    )


field_to_recommender = {
    "empathy": EmpathyRecommender,
    "problems_addressed": ProblemAddressingRecommender,
    "grammatical_errors": GrammaticalErrorsRecommender,
    "abbreviations": AbbreviationsRecommender,
    "punctuation_errors": PunctuationErrorsRecommender,
    "clarifications": ClarificationRecommender,
    "inside_questions": QuestionInsideResponseRecommender,
    "treatment_did_offer": TreatmentOfferedRecommender,
    "prescription_should_offer": PrescriptionRecommender,
    "explanation_causes": CausesExplanationRecommender,
    "explanation_symptoms": SymptomsExplanationRecommender,
    "explanation_treatment": TreatmentExplanationRecommender,
    "explanation_risk_factors": RiskFactorsExplanationRecommender,
    "explanation_next_steps": NextStepsExplanationRecommender,
    "generated_with_chatgpt": ChatGPTDetectorRecommender,
    "other_specialty": SpecialtyReferralRecommender,
    "only_recommends_visit": VisitOnlyRecommender,
    "cannot_help_online": OnlineHelpLimitationRecommender,
}


class RecommenderModule(dspy.Module):

    def __init__(self, cfg):
        super().__init__()
        self.recommenders = {}
        for field, signature in field_to_recommender.items():
            self.recommenders[field] = dspy.Predict(signature)

    async def aforward(
        self,
        scores: dict,
        patient_question: str,
        doctor_response: str,
        fields: Literal["all"] | List[str] = "all",
        max_tasks: int = 3,
        lm=None,
    ):

        if fields == "all":
            tasks = []
            fields_to_process = []

            for field_name, instance in self.recommenders.items():
                if field_name in scores and check_for_needed_recommendation(
                    field_name, scores
                ):
                    tasks.append(
                        instance.aforward(
                            patient_question=patient_question,
                            doctor_response=doctor_response,
                            score=scores[field_name],
                            lm=lm,
                        )
                    )
                    fields_to_process.append(field_name)

            results_list = await asyncio.gather(*tasks) if tasks else []

            results = {field_name: None for field_name in self.recommenders.keys()}
            for field_name, result in zip(fields_to_process, results_list):
                results[field_name] = result.recommendation

            return results

        # For single field case
        tasks = []
        fields_to_process = []
        for f in fields:
            if f in self.recommenders and check_for_needed_recommendation(f, scores):
                tasks.append(
                    self.recommenders[f].aforward(
                        patient_question=patient_question,
                        doctor_response=doctor_response,
                        score=scores[f],
                        lm=lm,
                    )
                )
                fields_to_process.append(f)
            if len(tasks) >= max_tasks:
                break

        results_list = await asyncio.gather(*tasks) if tasks else []

        results = {field: None for field in self.recommenders.keys()}
        for field_name, result in zip(fields_to_process, results_list):
            results[field_name] = result.recommendation

        return results

    def forward(
        self,
        scores: dict,
        patient_question: str,
        doctor_response: str,
        fields: Literal["all"] | List[str] = "all",
    ):
        if fields == "all":
            results = {}
            for field, instance in self.recommenders.items():
                if check_for_needed_recommendation(fields, scores):
                    results[field] = instance(
                        patient_question=patient_question,
                        doctor_response=doctor_response,
                        score=scores[field],
                    ).recommendation
                else:
                    results[field] = None
            return results

        for f in fields:
            results = {}
            if f in self.recommenders and check_for_needed_recommendation(f, scores):
                results[f] = self.recommenders[f](
                    patient_question=patient_question,
                    doctor_response=doctor_response,
                    score=scores[f],
                ).recommendation
            else:
                results[f] = None
            return results
        return {}
