from fpdf import FPDF

def generate_report(stock_data, stock_name, top_picks, suggested_sells):
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(200, 10, f"{stock_name} Performance Report", ln=True, align='C')

    # Summary
    pdf.set_font('Arial', '', 12)
    pdf.cell(200, 10, "Top 3 Stocks to Buy:", ln=True)
    for pick in top_picks:
        pdf.cell(200, 10, f" - {pick}", ln=True)

    pdf.cell(200, 10, f"Suggested Prices to Sell: {suggested_sells}", ln=True)

    # Save PDF
    pdf.output(f"../reports/{stock_name}_performance_report.pdf")

# Example usage:
# generate_report(stock_data, 'AAPL', ['AAPL', 'GOOGL', 'MSFT'], '230-250 USD')

