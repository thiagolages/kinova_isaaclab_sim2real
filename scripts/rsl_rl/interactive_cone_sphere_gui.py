import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QSlider, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QGridLayout
)
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import pyqtgraph.opengl as gl

def create_sphere_mesh(center, radius, color=(0, 0, 1, 0.3)):
    md = gl.MeshData.sphere(rows=30, cols=60, radius=radius)
    mesh = gl.GLMeshItem(
        meshdata=md,
        smooth=True,
        color=color,
        shader='shaded',
        drawFaces=True,
        drawEdges=False,
        glOptions='translucent'
    )
    mesh.translate(*center)
    return mesh

def create_cone_mesh(base_center, height, radius, color=(0, 1, 0, 0.3)):
    # Cone along z, base at base_center, height can be negative
    n_theta = 60
    theta = np.linspace(0, 2 * np.pi, n_theta)
    z_base = 0
    z_tip = height
    # Base circle
    x_base = base_center[0] + radius * np.cos(theta)
    y_base = base_center[1] + radius * np.sin(theta)
    z_base_arr = np.full_like(x_base, base_center[2])
    # Tip
    x_tip = base_center[0]
    y_tip = base_center[1]
    z_tip_val = base_center[2] + height
    # Vertices
    vertices = np.vstack([
        np.column_stack([x_base, y_base, z_base_arr]),
        [x_tip, y_tip, z_tip_val]
    ])
    # Faces
    faces = []
    tip_idx = len(vertices) - 1
    for i in range(n_theta - 1):
        faces.append([i, i + 1, tip_idx])
    faces.append([n_theta - 1, 0, tip_idx])
    faces = np.array(faces)
    md = gl.MeshData(vertexes=vertices, faces=faces)
    mesh = gl.GLMeshItem(
        meshdata=md,
        smooth=True,
        color=color,
        shader='shaded',
        drawFaces=True,
        drawEdges=False,
        glOptions='translucent'
    )
    return mesh

def create_center_sphere(center, radius=0.07, color=(1, 0, 0, 1.0)):
    md = gl.MeshData.sphere(rows=15, cols=30, radius=radius)
    mesh = gl.GLMeshItem(
        meshdata=md,
        smooth=True,
        color=color,
        shader='shaded',
        drawFaces=True,
        drawEdges=False,
        glOptions='opaque'
    )
    mesh.translate(*center)
    return mesh

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive 3D Cone and Sphere (PyQt5)")
        self.resize(900, 700)

        # 3D view
        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=4)
        self.view.setBackgroundColor('w')
        grid = gl.GLGridItem()
        grid.setSize(4, 4)
        grid.setSpacing(0.2, 0.2)
        self.view.addItem(grid)

        # Sliders
        self.sliders = {}
        self.labels = {}
        slider_params = [
            ('Sphere X', -1.5, 1.5, 0.0),
            ('Sphere Y', -1.5, 1.5, 0.0),
            ('Sphere Z', -1.5, 1.5, 0.0),
            ('Sphere R', 0.01, 1.5, 0.05),
            ('Cone X', -1.5, 1.5, 0.0),
            ('Cone Y', -1.5, 1.5, 0.0),
            ('Cone Z', -1.5, 1.5, 0.0),
            ('Cone H', -2.0, 2.0, 2.0),
            ('Cone R', 0.05, 1.5, 0.5),
        ]
        layout = QGridLayout()
        for i, (name, minv, maxv, val) in enumerate(slider_params):
            label = QLabel(f"{name}: {val:.2f}")
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(1000)
            slider.setValue(int((val - minv) / (maxv - minv) * 1000))
            slider.valueChanged.connect(self.update_plot)
            self.sliders[name] = (slider, minv, maxv)
            self.labels[name] = label
            layout.addWidget(label, i, 0)
            layout.addWidget(slider, i, 1)

        # Main layout
        vbox = QVBoxLayout()
        vbox.addWidget(self.view)
        vbox.addLayout(layout)
        container = QWidget()
        container.setLayout(vbox)
        self.setCentralWidget(container)

        # Initial plot
        self.update_plot()

    def get_slider_value(self, name):
        slider, minv, maxv = self.sliders[name]
        val = slider.value() / 1000 * (maxv - minv) + minv
        return val

    def update_plot(self, *args):
        # Remove old meshes
        for item in self.view.items[:]:
            if isinstance(item, gl.GLMeshItem):
                self.view.removeItem(item)
        # Get values
        sphere_x = self.get_slider_value('Sphere X')
        sphere_y = self.get_slider_value('Sphere Y')
        sphere_z = self.get_slider_value('Sphere Z')
        sphere_r = self.get_slider_value('Sphere R')
        cone_x = self.get_slider_value('Cone X')
        cone_y = self.get_slider_value('Cone Y')
        cone_z = self.get_slider_value('Cone Z')
        cone_h = self.get_slider_value('Cone H')
        cone_r = self.get_slider_value('Cone R')
        # Update labels
        for name in self.labels:
            val = self.get_slider_value(name)
            self.labels[name].setText(f"{name}: {val:.2f}")
        # Add meshes
        self.view.addItem(create_sphere_mesh([sphere_x, sphere_y, sphere_z], sphere_r))
        self.view.addItem(create_cone_mesh([cone_x, cone_y, cone_z], cone_h, cone_r))
        self.view.addItem(create_center_sphere([sphere_x, sphere_y, sphere_z], radius=0.07))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())