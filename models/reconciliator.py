import dspy
from configs.base import Config


class ReconciliatorSignature(dspy.Signature):
    """Improves the doctor's response using the provided recommendations"""

    patient_question = dspy.InputField(desc="Întrebarea pacientului")
    doctor_response = dspy.InputField(desc="Răspunsul doctorului, care va fi evaluat")
    recommendations = dspy.InputField(
        desc=f"Recomandări pentru a îmbunătăți răspunsul doctorului"
    )

    modified_response = dspy.OutputField(
        desc=f"Răspunsul doctorului modificat folosind recomandările"
    )


class ReconciliatorModule(dspy.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        self.reconciliator = dspy.Predict(ReconciliatorSignature)
        
    async def aforward(
        self, patient_question: str, doctor_response: str, recommendations: dict, lm=None
    ):
        output = await self.reconciliator.aforward(
            patient_question=patient_question,
            doctor_response=doctor_response,
            recommendations=recommendations,
            lm=lm
        )
        return output.modified_response

    def forward(
        self, patient_question: str, doctor_response: str, recommendations: dict
    ):
        return self.reconciliator(
            patient_question=patient_question,
            doctor_response=doctor_response,
            recommendations=recommendations,
        ).modified_response
