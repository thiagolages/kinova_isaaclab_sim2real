import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Initial parameters
sphere_center = [0, 0, 0]
sphere_radius = 1.0
cone_base_center = [0, 0, 0]
cone_height = 2.0
cone_radius = 0.5

def create_sphere(center, radius, color='rgba(0,0,255,0.3)'):
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    return go.Surface(
        x=x, y=y, z=z,
        opacity=0.3,
        colorscale=[[0, color], [1, color]],
        showscale=False,
        name='Sphere'
    )

def create_cone(base_center, height, radius, color='rgba(0,255,0,0.3)'):
    # Create a cone along the z-axis
    theta = np.linspace(0, 2 * np.pi, 40)
    z = np.linspace(0, height, 20)
    theta, z = np.meshgrid(theta, z)
    x = base_center[0] + (radius * (1 - z / height)) * np.cos(theta)
    y = base_center[1] + (radius * (1 - z / height)) * np.sin(theta)
    z = base_center[2] + z
    return go.Surface(
        x=x, y=y, z=z,
        opacity=0.3,
        colorscale=[[0, color], [1, color]],
        showscale=False,
        name='Cone'
    )

def create_center_sphere(center, radius=0.05, color='red'):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    return go.Surface(
        x=x, y=y, z=z,
        opacity=1.0,
        colorscale=[[0, color], [1, color]],
        showscale=False,
        name='Center'
    )

# Create initial plot
fig = go.Figure()

# Add sphere
fig.add_trace(create_sphere(sphere_center, sphere_radius))
# Add cone
fig.add_trace(create_cone(cone_base_center, cone_height, cone_radius))
# Add small red sphere at center
fig.add_trace(create_center_sphere(sphere_center, radius=0.07))

# Set up axes and grid
fig.update_layout(
    scene=dict(
        xaxis=dict(nticks=10, range=[-2, 2], backgroundcolor="white", gridcolor="lightgray", showbackground=True, title='X'),
        yaxis=dict(nticks=10, range=[-2, 2], backgroundcolor="white", gridcolor="lightgray", showbackground=True, title='Y'),
        zaxis=dict(nticks=10, range=[-2, 2], backgroundcolor="white", gridcolor="lightgray", showbackground=True, title='Z'),
        aspectmode='cube'
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    title="Interactive 3D Cone and Sphere"
)

# Add sliders for parameters
from ipywidgets import interact, FloatSlider, VBox
import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = "notebook_connected"  # or "jupyterlab" if using JupyterLab

def update_plot(
    sphere_x=0, sphere_y=0, sphere_z=0, sphere_r=1.0,
    cone_x=0, cone_y=0, cone_z=0, cone_h=2.0, cone_r=0.5
):
    fig = go.Figure()
    fig.add_trace(create_sphere([sphere_x, sphere_y, sphere_z], sphere_r))
    fig.add_trace(create_cone([cone_x, cone_y, cone_z], cone_h, cone_r))
    fig.add_trace(create_center_sphere([sphere_x, sphere_y, sphere_z], radius=0.07))
    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=10, range=[-2, 2], backgroundcolor="white", gridcolor="lightgray", showbackground=True, title='X'),
            yaxis=dict(nticks=10, range=[-2, 2], backgroundcolor="white", gridcolor="lightgray", showbackground=True, title='Y'),
            zaxis=dict(nticks=10, range=[-2, 2], backgroundcolor="white", gridcolor="lightgray", showbackground=True, title='Z'),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        title="Interactive 3D Cone and Sphere"
    )
    fig.show()

# Use ipywidgets for interactivity (works in Jupyter)
try:
    interact(
        update_plot,
        sphere_x=FloatSlider(min=-1.5, max=1.5, step=0.05, value=0, description='Sphere X'),
        sphere_y=FloatSlider(min=-1.5, max=1.5, step=0.05, value=0, description='Sphere Y'),
        sphere_z=FloatSlider(min=-1.5, max=1.5, step=0.05, value=0, description='Sphere Z'),
        sphere_r=FloatSlider(min=0.1, max=1.5, step=0.05, value=1.0, description='Sphere R'),
        cone_x=FloatSlider(min=-1.5, max=1.5, step=0.05, value=0, description='Cone X'),
        cone_y=FloatSlider(min=-1.5, max=1.5, step=0.05, value=0, description='Cone Y'),
        cone_z=FloatSlider(min=-1.5, max=1.5, step=0.05, value=0, description='Cone Z'),
        cone_h=FloatSlider(min=0.1, max=2.0, step=0.05, value=2.0, description='Cone H'),
        cone_r=FloatSlider(min=0.05, max=1.5, step=0.05, value=0.5, description='Cone R'),
    )
except Exception as e:
    print("Interactive sliders require a Jupyter environment (Jupyter Notebook or JupyterLab).")
    print("You can still view the static plot below.")
    fig.show()