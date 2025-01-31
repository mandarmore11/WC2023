import streamlit as st
from worldcup_utils import (load_data, plot_matrix_chart, generate_match_results,
                            order_teams_by_points_and_nrr, get_index_for_preselection,
                            get_team_points, calculate_nrr, 
                            generate_results_matrix, generate_standings_and_plot,
                            get_country_data, plot_country_graph,
                            get_victory_margin,add_line, plot_decagon)

def main(df, countries):
    st.title('World Cup 2023 Results')
    st.write('Choose options to see desired details')
    
    # Sidebar
    #st.sidebar.header('Options')
    # Add more options and widgets if needed


    
    choice = st.radio(
        "Choose a view:",
        ("Overall Stats", "Per Country Analysis")
    )

    if choice == "Overall Stats":
        # ... display overall statistics ...
        st.write("Overall statistics go here.")

        # Create a row with three columns
        col1, col2, col3 = st.columns(3)
        # Place a checkbox in each column
        with col1:
            checkbox1 = st.checkbox("Standings and Victories", value=True)
        with col2:
            checkbox2 = st.checkbox("Standings (sorted)")
        with col3:
            checkbox3 = st.checkbox("Per match details")

        if checkbox1:
            st.write('Who beat whom')
            plot_decagon(df)

        if checkbox2:
            st.write('Team standings')
            generate_standings_and_plot(df)

        # Show Raw Data
        if checkbox3:
            st.write('Per match details')
            st.write(df)

    elif choice == "Per Country Analysis":
        # Create a row with columns
        col11, col12 = st.columns(2)
        # Place a checkbox in each column
        with col11:
            checkbox11 = st.checkbox("Match details")
        with col12:
            checkbox12 = st.checkbox("Graphic", value=True)
        # Dropdown for country selection
        index_to_preselect = get_index_for_preselection(countries, 'Ind')
        selected_country = st.selectbox('Choose a country:', countries,index=index_to_preselect)
        country_specific_df = get_country_data(df,selected_country)
        if checkbox11:
            st.write('All the rows')
            st.write(country_specific_df)
        if checkbox12:
            st.write('Who beat whom')
            plot_country_graph(selected_country, country_specific_df, countries)
    
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

    # Divider
    st.write("---")

    # Source Code
    st.write("**Source Code:** [GitHub Repo](https://github.com/AshishMahabal/WC2023) PRs welcome.")

    # Version
    st.write("**Version:** 0.0.1")

    # Disclaimer
    st.write("### Disclaimer")
    st.write("This app makes no guarantees - use at your own peril.")

# Run Streamlit app
if __name__ == '__main__':
    filename = "worldcup2023results.csv"
    df = load_data(filename)
#    generate_standings_and_plot(df)
#    data = load_data()
    countries = ["Ind", "Aus", "NZ", "SL", "Ban", "Pak", "Eng", "SA", "Ned", "Afg"]
    main(df, countries)

