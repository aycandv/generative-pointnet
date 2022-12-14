import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots


def visualize_subplots(x, y):
    """Visualize point cloud

    Args:
        fig_train list(go.Figure): List of go.Figure objects
        fig_pred list(go.Figure): List of go.Figure objects

    Returns:
        fig (go.Figure): Visualized point cloud
    """
    num_samples = 5 if x.shape[0] > 5 else x.shape[0]
    fig = make_subplots(
        rows=2,
        cols=num_samples,
        specs=[
            [{"type": "scene", "showticklabels": False} for _ in range(num_samples)],
            [{"type": "scene", "showticklabels": False} for _ in range(num_samples)],
        ],
    )

    for i in range(num_samples):
        fig_train = pcshow(x[i, :, 0].unsqueeze(0), x[i, :, 1].unsqueeze(0), x[i, :, 2].unsqueeze(0))
        fig_pred = pcshow(y[i, :, 0].unsqueeze(0), y[i, :, 1].unsqueeze(0), y[i, :, 2].unsqueeze(0))
        for t in fig_train.data:
            fig.add_trace(t, row=1, col=i+1)
        for p in fig_pred.data:
            fig.add_trace(p, row=2, col=i+1)

    return fig


def visualize_rotate(data):
    """Visualize rotating 3D point cloud

    Args:
        data (list): List of go.Scatter3d objects

    Returns:
        fig (go.Figure): Rotating 3D point cloud
    """
    x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
    frames = []

    def rotate_z(x, y, z, theta):
        w = x + 1j * y
        return np.real(np.exp(1j * theta) * w), np.imag(np.exp(1j * theta) * w), z

    for t in np.arange(0, 10.26, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(
            dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze)))))
        )
    fig = go.Figure(
        data=data,
        layout=go.Layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=1,
                    x=0.8,
                    xanchor="left",
                    yanchor="bottom",
                    pad=dict(t=45, r=10),
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=50, redraw=True),
                                    transition=dict(duration=0),
                                    fromcurrent=True,
                                    mode="immediate",
                                ),
                            ],
                        )
                    ],
                )
            ]
        ),
        frames=frames,
    )

    return fig


def pcshow(xs, ys, zs):
    """Visualize point cloud

    Args:
        xs (np.ndarray): x coordinates
        ys (np.ndarray): y coordinates
        zs (np.ndarray): z coordinates
    """
    data = [go.Scatter3d(x=x, y=y, z=z, mode="markers") for x, y, z in zip(xs, ys, zs)]
    fig = visualize_rotate(data)
    fig.update_traces(
        marker=dict(size=2, line=dict(width=2, color="DarkSlateGrey")),
        selector=dict(mode="markers"),
    )
    return fig
