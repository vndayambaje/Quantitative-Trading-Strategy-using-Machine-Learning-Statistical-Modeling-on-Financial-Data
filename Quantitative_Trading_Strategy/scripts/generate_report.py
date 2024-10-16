from fpdf import FPDF
import pandas as pd
import os

def generate_report(stock_data, predictions, stock_name, initial_balance, final_balance):
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(200, 10, f"{stock_name} Performance Report", ln=True, align='C')

    # Summary
    pdf.set_font('Arial', '', 12)
    pdf.cell(200, 10, f"Initial Balance: ${initial_balance}", ln=True)
    pdf.cell(200, 10, f"Final Balance: ${final_balance:.2f}", ln=True)

    # Table header
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(40, 10, 'Date', 1)
    pdf.cell(40, 10, 'Close Price', 1)
    pdf.cell(40, 10, 'Prediction', 1)
    pdf.cell(40, 10, 'Holding Period', 1)
    pdf.ln()

    # Table content
    for index, row in stock_data.tail(5).iterrows():
        pdf.cell(40, 10, str(index.date()), 1)
        pdf.cell(40, 10, f"${row['Close']:.2f}", 1)
        pdf.cell(40, 10, f"{predictions.loc[index]:.2f}", 1)
        pdf.cell(40, 10, "5 days", 1)  # Example holding period
        pdf.ln()

    # Define the file path
    output_path = os.path.join('/home/vic3/github/Quantitative-Trading-Strategy-using-Machine-Learning-Statistical-Modeling-on-Financial-Data/Quantitative_Trading_Strategy/data', f"{stock_name}_performance_report.pdf")

    # Save PDF in the data directory
    pdf.output(output_path)
    print(f"Report saved at: {output_path}")

