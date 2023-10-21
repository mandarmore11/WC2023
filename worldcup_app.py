import streamlit as st
from worldcup_utils import (load_data, plot_matrix_chart, generate_match_results,
                            order_teams_by_points_and_nrr, 
                            get_team_points, calculate_nrr, 
                            generate_results_matrix, generate_standings_and_plot,
                            get_victory_margin,add_line, plot_decagon)

def main():
    st.title('World Cup 2023 Results')
    
    # Sidebar
    st.sidebar.header('Options')
    # Add more options and widgets if needed
    
    # Show Raw Data
    if st.sidebar.checkbox('Show Raw Data', False):
        st.write(df)
    if st.sidebar.checkbox('Show Standing', False):
        generate_standings_and_plot(df)
    if st.sidebar.checkbox('Show Decagon', False):
        plot_decagon(df)
#        st.write(df)
    
    # Compute matrix
#    team_points, team_nrr = compute_team_metrics(data, countries)
    # Get points and NRR for sorting
    team_points = get_team_points(df, countries)
    team_nrr = calculate_nrr(df, countries)
    sorted_countries = sorted(countries, key=lambda x: (team_points[x], team_nrr[x]), reverse=True)
#    matrix = create_match_matrix(data, sorted_countries)
    results_matrix = generate_results_matrix(df, sorted_countries)
    
    # Plot matrix chart
    plot_matrix_chart(results_matrix, sorted_countries, team_points, team_nrr, df)
    
    # Any other visualizations or features can be added here...

# Run Streamlit app
if __name__ == '__main__':
    filename = "worldcup2023results.csv"
    df = load_data(filename)
#    generate_standings_and_plot(df)
#    data = load_data()
    countries = ["Aus", "Eng", "SA", "NZ", "Pak", "Ind", "SL", "Ban", "Afg", "Ned"]
    main()

