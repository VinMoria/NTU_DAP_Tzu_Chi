import sqlite3
import pandas as pd
import json

def create_table():
    conn = sqlite3.connect("Tzuchi.db")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS case_profile (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            assessment_date TEXT NOT NULL,
            type_of_assistances TEXT NOT NULL,
            age INTEGER NOT NULL,
            occupation TEXT NOT NULL,
            intake_no_of_hh INTEGER NOT NULL,

            -- Income fields
            income_assessment_salary REAL NOT NULL,
            income_assessment_cpf_payout REAL NOT NULL,
            income_assessment_assistance_from_other_agencies REAL NOT NULL,
            income_assessment_assistance_from_relatives_friends REAL NOT NULL,
            income_assessment_insurance_payout REAL NOT NULL,
            income_assessment_rental_income REAL NOT NULL,
            income_assessment_others_income REAL NOT NULL,
            current_savings REAL NOT NULL,

            -- Expenditure fields
            expenditure_assessment_mortgage_rental REAL NOT NULL,
            expenditure_assessment_utilities REAL NOT NULL,
            expenditure_assessment_s_cc_fees REAL NOT NULL,
            expenditure_assessment_food_expenses REAL NOT NULL,
            expenditure_assessment_marketing_groceries REAL NOT NULL,
            expenditure_assessment_telecommunications REAL NOT NULL,
            expenditure_assessment_transportation REAL NOT NULL,
            expenditure_assessment_medical_expenses REAL NOT NULL,
            expenditure_assessment_education_expense REAL NOT NULL,
            expenditure_assessment_contribution_to_family_members REAL NOT NULL,
            expenditure_assessment_domestic_helper REAL NOT NULL,
            expenditure_assessment_loans_debts_installments REAL NOT NULL,
            expenditure_assessment_insurance_premiums REAL NOT NULL,
            expenditure_assessment_others_expenditure REAL NOT NULL,

            -- HH fields
            no_of_hh INTEGER NOT NULL,
            before_primary INTEGER NOT NULL,
            primary_7_12 INTEGER NOT NULL,
            secondary_13_17 INTEGER NOT NULL,
            tertiary_18_21 INTEGER NOT NULL,
            adult_22_64 INTEGER NOT NULL,
            elderly_65_and_above INTEGER NOT NULL,
            points IINTEGER NOT NULL,

            -- Y value
            amount_total REAL NOT NULL,

            -- feedback
            feedback_val REAL
        )
        """
    )
    conn.close()



def insert_case_data(df):
    # Make sure the DataFrame has the expected structure
    expected_columns = [
        'assessment_date', 
        'type_of_assistances', 
        'age', 
        'occupation', 
        'intake_no_of_hh',
        'income_assessment_salary', 
        'income_assessment_cpf_payout',
        'income_assessment_assistance_from_other_agencies', 
        'income_assessment_assistance_from_relatives_friends', 
        'income_assessment_insurance_payout', 
        'income_assessment_rental_income', 
        'income_assessment_others_income', 
        'current_savings',
        'expenditure_assessment_mortgage_rental', 
        'expenditure_assessment_utilities',
        'expenditure_assessment_s_cc_fees', 
        'expenditure_assessment_food_expenses',
        'expenditure_assessment_marketing_groceries', 
        'expenditure_assessment_telecommunications',
        'expenditure_assessment_transportation', 
        'expenditure_assessment_medical_expenses',
        'expenditure_assessment_education_expense', 
        'expenditure_assessment_contribution_to_family_members',
        'expenditure_assessment_domestic_helper', 
        'expenditure_assessment_loans_debts_installments',
        'expenditure_assessment_insurance_premiums', 
        'expenditure_assessment_others_expenditure',
        'no_of_hh', 
        'before_primary', 
        'primary_7_12', 
        'secondary_13_17', 
        'tertiary_18_21',
        'adult_22_64', 
        'elderly_65_and_above',
        'points',
        'amount_total'
    ]

    # Check if DataFrame has the required columns
    if not all(col in df.columns for col in expected_columns):
        raise ValueError(f"Input DataFrame must contain columns: {expected_columns}")

    # Convert DataFrame to dictionary for easier access
    data_row = df.iloc[0].to_dict()

    # Database insertion
    conn = sqlite3.connect("Tzuchi.db")
    cur = conn.cursor()
    
    cur.execute(f"""
        INSERT INTO case_profile ({', '.join(expected_columns)}) VALUES ({', '.join(['?']*len(expected_columns))})
    """, [data_row[col] for col in expected_columns])

    # Get the ID of the newly inserted row
    new_id = cur.lastrowid
    conn.commit()
    conn.close()

    return new_id

# Example usage:
# Assuming `df` is your DataFrame with the required structure
# id = insert_case_data(df)

def update_feedback(case_id, feedback_val):
    
    conn = sqlite3.connect("Tzuchi.db")
    cur = conn.cursor()
    
    cur.execute("""
        UPDATE case_profile
        SET feedback_val = ?
        WHERE id = ?
    """, (feedback_val, case_id))
    
    if cur.rowcount == 0:
        print(f"No record found with id: {case_id}.")
    
    conn.commit()
    conn.close()

def select_all_cases():
    conn = sqlite3.connect("Tzuchi.db")
    cur = conn.cursor()

    cur.execute("SELECT * FROM case_profile")
    rows = cur.fetchall()

    conn.close()
    return rows

def select_case_by_id(profile_id):

    conn = sqlite3.connect("Tzuchi.db")
    cur = conn.cursor()

    cur.execute("SELECT * FROM case_profile WHERE id = ?", (profile_id,))
    row = cur.fetchone()

    if row is None:
        conn.close()
        return None

    # 获取列名
    cur.execute("SELECT * FROM case_profile")
    column_names = [description[0] for description in cur.description]

    # 关闭数据库连接
    conn.close()

    # 从行数据创建字典
    data_dict = dict(zip(column_names, row))

    return data_dict

if __name__ == "__main__":
    create_table()