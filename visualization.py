import plotly.express as px

def plot_pulse(x, y, prediction):
    """Interactive pulse visulization with Plotly"""
    fig = px.line(x=x, y=y, title= "Noisy Pulse with Predictions")
    fig.add_vline(x= prediction[0], line_dash= "dash", line_color= "red", name= "Predicted Peak")
    fig.add_vline(x= prediction[1], line_dash= "dash", line_color= "blue", name= "Left Threshold")
    fig.add_vline(x= prediction[2], line_dash= "dash", line_color= "green", name= "Right Threshold")
    fig.show()