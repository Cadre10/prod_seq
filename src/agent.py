from src.io_data import load_daily_plan, load_form_responses

def main():
    daily_path = "data/_Clandeboye Production Perfomance dashboard - Daily_Plan.csv"
    form_path = "data/_Clandeboye Production Perfomance dashboard - Form_Responses.csv"

    df_daily = load_daily_plan(daily_path)
    df_form = load_form_responses(form_path)

    print(df_daily.head(3))
    print(df_form.head(3))

if __name__ == "__main__":
    main()
