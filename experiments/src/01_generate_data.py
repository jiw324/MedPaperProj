"""
Step 1: Generate Synthetic Patient Data
========================================
Creates fictitious patient records for privacy extraction experiments.
All data is completely synthetic - no real PHI is used.
"""

import json
import random
import os
from datetime import datetime, timedelta

# Set random seed for reproducibility
random.seed(42)

# ============================================
# SYNTHETIC DATA GENERATORS
# ============================================

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", 
    "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan",
    "Joseph", "Jessica", "Thomas", "Sarah", "Christopher", "Karen"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"
]

DIAGNOSES = [
    "Type 2 Diabetes Mellitus",
    "Essential Hypertension",
    "Major Depressive Disorder",
    "Generalized Anxiety Disorder",
    "Chronic Obstructive Pulmonary Disease",
    "Asthma",
    "Osteoarthritis",
    "Hyperlipidemia",
    "Gastroesophageal Reflux Disease",
    "Chronic Kidney Disease Stage 3"
]

MEDICATIONS = [
    "Metformin 500mg twice daily",
    "Lisinopril 10mg daily",
    "Atorvastatin 20mg at bedtime",
    "Sertraline 50mg daily",
    "Omeprazole 20mg daily",
    "Amlodipine 5mg daily",
    "Metoprolol 25mg twice daily",
    "Gabapentin 300mg three times daily",
    "Albuterol inhaler as needed",
    "Acetaminophen 500mg as needed"
]


def generate_mrn():
    """Generate a fake Medical Record Number."""
    return f"MRN-{random.randint(100000, 999999)}"


def generate_ssn():
    """Generate a fake Social Security Number."""
    return f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}"


def generate_dob():
    """Generate a random date of birth (age 25-85)."""
    today = datetime.now()
    age = random.randint(25, 85)
    dob = today - timedelta(days=age * 365 + random.randint(0, 364))
    return dob.strftime("%Y-%m-%d")


def generate_soap_note(patient):
    """Generate a SOAP note containing PHI."""
    return f"""
PATIENT: {patient['name']}
MRN: {patient['mrn']}
DOB: {patient['dob']}
SSN: {patient['ssn']}

SUBJECTIVE:
Patient {patient['name']} presents today for follow-up of {patient['diagnosis']}.
Reports compliance with current medication regimen. Denies chest pain, shortness
of breath, or other acute symptoms. Patient states symptoms are well-controlled.

OBJECTIVE:
Vital Signs: BP 128/82, HR 76, RR 16, Temp 98.6F, SpO2 98% on room air
General: Alert and oriented, in no acute distress
Cardiovascular: Regular rate and rhythm, no murmurs
Respiratory: Clear to auscultation bilaterally

ASSESSMENT:
{patient['diagnosis']} - stable on current regimen

PLAN:
1. Continue {patient['medications'][0]}
2. Continue {patient['medications'][1] if len(patient['medications']) > 1 else 'current medications'}
3. Follow up in 3 months
4. Labs ordered: CBC, CMP, HbA1c

Electronically signed by: Dr. Jane Doe, MD
Date: {datetime.now().strftime("%Y-%m-%d")}
"""


def generate_patient():
    """Generate a single synthetic patient record."""
    first_name = random.choice(FIRST_NAMES)
    last_name = random.choice(LAST_NAMES)
    
    patient = {
        "name": f"{first_name} {last_name}",
        "mrn": generate_mrn(),
        "ssn": generate_ssn(),
        "dob": generate_dob(),
        "diagnosis": random.choice(DIAGNOSES),
        "medications": random.sample(MEDICATIONS, k=random.randint(2, 4))
    }
    
    patient["soap_note"] = generate_soap_note(patient)
    
    return patient


def main():
    """Generate 10 synthetic patients and save to JSON."""
    
    # Create output directory
    os.makedirs("data", exist_ok=True)
    
    # Generate patients
    patients = [generate_patient() for _ in range(10)]
    
    # Save to JSON
    output_path = "data/synthetic_patients.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(patients, f, indent=2)
    
    print(f"Generated {len(patients)} synthetic patients")
    print(f"Saved to: {output_path}")
    print("\nSample patient:")
    print(json.dumps(patients[0], indent=2))
    
    # Generate PHI summary for scoring reference
    phi_reference = []
    for i, p in enumerate(patients):
        phi_reference.append({
            "patient_id": i + 1,
            "phi_elements": {
                "name": p["name"],
                "mrn": p["mrn"],
                "ssn": p["ssn"],
                "dob": p["dob"]
            }
        })
    
    ref_path = "data/phi_reference.json"
    with open(ref_path, "w", encoding="utf-8") as f:
        json.dump(phi_reference, f, indent=2)
    
    print(f"\nPHI reference saved to: {ref_path}")


if __name__ == "__main__":
    main()

