import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QTabWidget, QLabel, QPushButton,
    QComboBox, QLineEdit, QGroupBox, QFormLayout, QSpinBox,
    QDoubleSpinBox, QTextEdit, QSplitter, QFrame, QMessageBox,
    QFileDialog, QHeaderView, QScrollArea, QGridLayout, QStatusBar,
    QAction, QToolBar,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.fig.tight_layout()


class DataExplorerTab(QWidget):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.filtered_df = df.copy()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Filter section
        filter_group = QGroupBox("Filters")
        filter_layout = QGridLayout()

        # Traffic level filter
        filter_layout.addWidget(QLabel("Traffic Level:"), 0, 0)
        self.traffic_combo = QComboBox()
        self.traffic_combo.addItems(["All"] + list(self.df['traffic_level'].unique()))
        self.traffic_combo.currentTextChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.traffic_combo, 0, 1)

        # Delivery mode filter
        filter_layout.addWidget(QLabel("Delivery Mode:"), 0, 2)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["All"] + list(self.df['delivery_mode'].unique()))
        self.mode_combo.currentTextChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.mode_combo, 0, 3)

        # Weather filter
        filter_layout.addWidget(QLabel("Weather:"), 0, 4)
        self.weather_combo = QComboBox()
        self.weather_combo.addItems(["All"] + list(self.df['weather'].unique()))
        self.weather_combo.currentTextChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.weather_combo, 0, 5)

        # Restaurant zone filter
        filter_layout.addWidget(QLabel("Restaurant Zone:"), 1, 0)
        self.rest_zone_combo = QComboBox()
        self.rest_zone_combo.addItems(["All"] + list(self.df['restaurant_zone'].unique()))
        self.rest_zone_combo.currentTextChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.rest_zone_combo, 1, 1)

        # Customer zone filter
        filter_layout.addWidget(QLabel("Customer Zone:"), 1, 2)
        self.cust_zone_combo = QComboBox()
        self.cust_zone_combo.addItems(["All"] + list(self.df['customer_zone'].unique()))
        self.cust_zone_combo.currentTextChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.cust_zone_combo, 1, 3)

        # Distance range
        filter_layout.addWidget(QLabel("Min Distance (km):"), 2, 0)
        self.min_dist = QDoubleSpinBox()
        self.min_dist.setRange(0, 100)
        self.min_dist.setValue(0)
        self.min_dist.valueChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.min_dist, 2, 1)

        filter_layout.addWidget(QLabel("Max Distance (km):"), 2, 2)
        self.max_dist = QDoubleSpinBox()
        self.max_dist.setRange(0, 100)
        self.max_dist.setValue(self.df['distance_km'].max())
        self.max_dist.valueChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.max_dist, 2, 3)

        # Reset button
        self.reset_btn = QPushButton("Reset Filters")
        self.reset_btn.clicked.connect(self.reset_filters)
        filter_layout.addWidget(self.reset_btn, 2, 5)

        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)

        # Search section
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search Order ID:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter order ID...")
        self.search_input.textChanged.connect(self.search_order)
        search_layout.addWidget(self.search_input)
        layout.addLayout(search_layout)

        # Record count label
        self.record_label = QLabel(f"Showing {len(self.df)} records")
        self.record_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(self.record_label)

        # Data table
        self.table = QTableWidget()
        self.setup_table()
        layout.addWidget(self.table)

        # Export button
        export_layout = QHBoxLayout()
        self.export_btn = QPushButton("Export Filtered Data to CSV")
        self.export_btn.clicked.connect(self.export_data)
        export_layout.addWidget(self.export_btn)
        export_layout.addStretch()
        layout.addLayout(export_layout)

        self.setLayout(layout)

    def setup_table(self):
        self.table.setColumnCount(len(self.df.columns))
        self.table.setHorizontalHeaderLabels(self.df.columns.tolist())
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        self.populate_table(self.df)

    def populate_table(self, data):
        self.table.setRowCount(len(data))
        for i, row in data.iterrows():
            for j, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(self.table.rowCount() - len(data) + list(data.index).index(i), j, item)
        self.record_label.setText(f"Showing {len(data)} records")

    def apply_filters(self):
        self.filtered_df = self.df.copy()

        if self.traffic_combo.currentText() != "All":
            self.filtered_df = self.filtered_df[self.filtered_df['traffic_level'] == self.traffic_combo.currentText()]

        if self.mode_combo.currentText() != "All":
            self.filtered_df = self.filtered_df[self.filtered_df['delivery_mode'] == self.mode_combo.currentText()]

        if self.weather_combo.currentText() != "All":
            self.filtered_df = self.filtered_df[self.filtered_df['weather'] == self.weather_combo.currentText()]

        if self.rest_zone_combo.currentText() != "All":
            self.filtered_df = self.filtered_df[self.filtered_df['restaurant_zone'] == self.rest_zone_combo.currentText()]

        if self.cust_zone_combo.currentText() != "All":
            self.filtered_df = self.filtered_df[self.filtered_df['customer_zone'] == self.cust_zone_combo.currentText()]

        self.filtered_df = self.filtered_df[
            (self.filtered_df['distance_km'] >= self.min_dist.value()) &
            (self.filtered_df['distance_km'] <= self.max_dist.value())
        ]

        self.populate_table(self.filtered_df)

    def reset_filters(self):
        self.traffic_combo.setCurrentText("All")
        self.mode_combo.setCurrentText("All")
        self.weather_combo.setCurrentText("All")
        self.rest_zone_combo.setCurrentText("All")
        self.cust_zone_combo.setCurrentText("All")
        self.min_dist.setValue(0)
        self.max_dist.setValue(self.df['distance_km'].max())
        self.search_input.clear()
        self.filtered_df = self.df.copy()
        self.populate_table(self.df)

    def search_order(self):
        search_text = self.search_input.text()
        if search_text:
            try:
                order_id = int(search_text)
                result = self.filtered_df[self.filtered_df['order_id'] == order_id]
                self.populate_table(result)
            except ValueError:
                self.populate_table(self.filtered_df)
        else:
            self.populate_table(self.filtered_df)

    def export_data(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
        if filename:
            self.filtered_df.to_csv(filename, index=False)
            QMessageBox.information(self, "Export Successful", f"Data exported to {filename}")


class StatisticsTab(QWidget):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Summary statistics section
        summary_group = QGroupBox("Summary Statistics")
        summary_layout = QVBoxLayout()

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setFont(QFont("Courier", 10))
        self.update_statistics()
        summary_layout.addWidget(self.stats_text)

        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)

        # Category selection for detailed stats
        detail_group = QGroupBox("Detailed Statistics by Category")
        detail_layout = QVBoxLayout()

        cat_layout = QHBoxLayout()
        cat_layout.addWidget(QLabel("Select Category:"))
        self.category_combo = QComboBox()
        self.category_combo.addItems(['traffic_level', 'delivery_mode', 'weather', 
                                      'restaurant_zone', 'customer_zone'])
        self.category_combo.currentTextChanged.connect(self.update_category_stats)
        cat_layout.addWidget(self.category_combo)
        cat_layout.addStretch()
        detail_layout.addLayout(cat_layout)

        self.category_stats_text = QTextEdit()
        self.category_stats_text.setReadOnly(True)
        self.category_stats_text.setFont(QFont("Courier", 10))
        self.update_category_stats()
        detail_layout.addWidget(self.category_stats_text)

        detail_group.setLayout(detail_layout)
        layout.addWidget(detail_group)

        self.setLayout(layout)

    def update_statistics(self):
        stats_text = "=" * 60 + "\n"
        stats_text += "DATASET OVERVIEW\n"
        stats_text += "=" * 60 + "\n\n"
        stats_text += f"Total Records: {len(self.df)}\n"
        stats_text += f"Total Columns: {len(self.df.columns)}\n\n"

        stats_text += "-" * 60 + "\n"
        stats_text += "NUMERICAL STATISTICS\n"
        stats_text += "-" * 60 + "\n\n"

        numeric_cols = ['distance_km', 'delivery_time_min', 'route_length_km']
        for col in numeric_cols:
            stats_text += f"{col}:\n"
            stats_text += f"  Mean: {self.df[col].mean():.2f}\n"
            stats_text += f"  Std: {self.df[col].std():.2f}\n"
            stats_text += f"  Min: {self.df[col].min():.2f}\n"
            stats_text += f"  Max: {self.df[col].max():.2f}\n"
            stats_text += f"  Median: {self.df[col].median():.2f}\n\n"

        stats_text += "-" * 60 + "\n"
        stats_text += "CATEGORICAL DISTRIBUTIONS\n"
        stats_text += "-" * 60 + "\n\n"

        cat_cols = ['traffic_level', 'delivery_mode', 'weather', 'restaurant_zone', 'customer_zone']
        for col in cat_cols:
            stats_text += f"{col}:\n"
            for val, count in self.df[col].value_counts().items():
                pct = count / len(self.df) * 100
                stats_text += f"  {val}: {count} ({pct:.1f}%)\n"
            stats_text += "\n"

        self.stats_text.setText(stats_text)

    def update_category_stats(self):
        category = self.category_combo.currentText()
        stats_text = f"Statistics grouped by {category}\n"
        stats_text += "=" * 60 + "\n\n"

        grouped = self.df.groupby(category).agg({
            'delivery_time_min': ['mean', 'std', 'min', 'max', 'count'],
            'distance_km': ['mean', 'std'],
            'route_length_km': ['mean', 'std']
        }).round(2)

        for idx in grouped.index:
            stats_text += f"{category}: {idx}\n"
            stats_text += "-" * 40 + "\n"
            stats_text += f"  Count: {int(grouped.loc[idx, ('delivery_time_min', 'count')])}\n"
            stats_text += f"  Delivery Time - Mean: {grouped.loc[idx, ('delivery_time_min', 'mean')]:.2f} min\n"
            stats_text += f"  Delivery Time - Std: {grouped.loc[idx, ('delivery_time_min', 'std')]:.2f} min\n"
            stats_text += f"  Distance - Mean: {grouped.loc[idx, ('distance_km', 'mean')]:.2f} km\n"
            stats_text += f"  Route Length - Mean: {grouped.loc[idx, ('route_length_km', 'mean')]:.2f} km\n\n"

        self.category_stats_text.setText(stats_text)


class VisualizationTab(QWidget):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Chart selection
        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("Select Chart:"))
        self.chart_combo = QComboBox()
        self.chart_combo.addItems([
            "Delivery Time Distribution",
            "Distance vs Delivery Time",
            "Delivery Time by Mode",
            "Delivery Time by Traffic",
            "Delivery Time by Weather",
            "Orders by Hour",
            "Zone Flow Heatmap",
            "Speed by Delivery Mode",
            "Distance Distribution",
            "Route Efficiency by Mode",
            "Correlation Heatmap",
            "Orders by Day of Week"
        ])
        self.chart_combo.currentTextChanged.connect(self.update_chart)
        control_layout.addWidget(self.chart_combo)
        control_layout.addStretch()

        self.refresh_btn = QPushButton("Refresh Chart")
        self.refresh_btn.clicked.connect(self.update_chart)
        control_layout.addWidget(self.refresh_btn)

        layout.addLayout(control_layout)

        # Canvas for matplotlib
        self.canvas = MplCanvas(self, width=10, height=7, dpi=100)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.setLayout(layout)
        self.update_chart()

    def update_chart(self):
        self.canvas.fig.clear()
        self.canvas.axes = self.canvas.fig.add_subplot(111)
        chart_type = self.chart_combo.currentText()

        if chart_type == "Delivery Time Distribution":
            self.canvas.axes.hist(self.df['delivery_time_min'], bins=30, 
                                  edgecolor='black', color='steelblue')
            self.canvas.axes.axvline(self.df['delivery_time_min'].mean(), 
                                     color='red', linestyle='--', 
                                     label=f'Mean: {self.df["delivery_time_min"].mean():.1f}')
            self.canvas.axes.set_xlabel('Delivery Time (min)')
            self.canvas.axes.set_ylabel('Frequency')
            self.canvas.axes.set_title('Distribution of Delivery Time')
            self.canvas.axes.legend()

        elif chart_type == "Distance vs Delivery Time":
            traffic_map = {'Low': 0, 'Medium': 1, 'High': 2}
            colors = self.df['traffic_level'].map(traffic_map)
            scatter = self.canvas.axes.scatter(self.df['distance_km'], 
                                               self.df['delivery_time_min'],
                                               c=colors, cmap='RdYlGn_r', alpha=0.6)
            self.canvas.axes.set_xlabel('Distance (km)')
            self.canvas.axes.set_ylabel('Delivery Time (min)')
            self.canvas.axes.set_title('Distance vs Delivery Time by Traffic Level')
            cbar = self.canvas.fig.colorbar(scatter, ax=self.canvas.axes)
            cbar.set_ticks([0, 1, 2])
            cbar.set_ticklabels(['Low', 'Medium', 'High'])

        elif chart_type == "Delivery Time by Mode":
            mode_data = self.df.groupby('delivery_mode')['delivery_time_min'].mean().sort_values()
            self.canvas.axes.bar(mode_data.index, mode_data.values, color='teal')
            self.canvas.axes.set_xlabel('Delivery Mode')
            self.canvas.axes.set_ylabel('Average Delivery Time (min)')
            self.canvas.axes.set_title('Average Delivery Time by Mode')

        elif chart_type == "Delivery Time by Traffic":
            traffic_order = ['Low', 'Medium', 'High']
            traffic_data = self.df.groupby('traffic_level')['delivery_time_min'].mean().reindex(traffic_order)
            colors = ['green', 'orange', 'red']
            self.canvas.axes.bar(traffic_data.index, traffic_data.values, color=colors)
            self.canvas.axes.set_xlabel('Traffic Level')
            self.canvas.axes.set_ylabel('Average Delivery Time (min)')
            self.canvas.axes.set_title('Impact of Traffic on Delivery Time')

        elif chart_type == "Delivery Time by Weather":
            weather_data = self.df.groupby('weather')['delivery_time_min'].mean().sort_values()
            self.canvas.axes.bar(weather_data.index, weather_data.values, 
                                color='skyblue', edgecolor='navy')
            self.canvas.axes.set_xlabel('Weather')
            self.canvas.axes.set_ylabel('Average Delivery Time (min)')
            self.canvas.axes.set_title('Impact of Weather on Delivery Time')

        elif chart_type == "Orders by Hour":
            df_temp = self.df.copy()
            df_temp['hour'] = pd.to_datetime(df_temp['order_time']).dt.hour
            hourly = df_temp.groupby('hour')['order_id'].count()
            self.canvas.axes.plot(hourly.index, hourly.values, marker='o', 
                                  linewidth=2, color='purple')
            self.canvas.axes.fill_between(hourly.index, hourly.values, alpha=0.3, color='purple')
            self.canvas.axes.set_xlabel('Hour of Day')
            self.canvas.axes.set_ylabel('Number of Orders')
            self.canvas.axes.set_title('Order Distribution by Hour')
            self.canvas.axes.set_xticks(range(0, 24, 2))

        elif chart_type == "Zone Flow Heatmap":
            zone_counts = pd.crosstab(self.df['restaurant_zone'], self.df['customer_zone'])
            im = self.canvas.axes.imshow(zone_counts.values, cmap='YlOrRd')
            self.canvas.axes.set_xticks(range(len(zone_counts.columns)))
            self.canvas.axes.set_yticks(range(len(zone_counts.index)))
            self.canvas.axes.set_xticklabels(zone_counts.columns)
            self.canvas.axes.set_yticklabels(zone_counts.index)
            self.canvas.axes.set_xlabel('Customer Zone')
            self.canvas.axes.set_ylabel('Restaurant Zone')
            self.canvas.axes.set_title('Order Flow Between Zones')
            for i in range(len(zone_counts.index)):
                for j in range(len(zone_counts.columns)):
                    self.canvas.axes.text(j, i, zone_counts.values[i, j],
                                         ha='center', va='center')
            self.canvas.fig.colorbar(im, ax=self.canvas.axes)

        elif chart_type == "Speed by Delivery Mode":
            df_temp = self.df.copy()
            df_temp['speed'] = (df_temp['distance_km'] / df_temp['delivery_time_min']) * 60
            speed_data = df_temp.groupby('delivery_mode')['speed'].mean().sort_values()
            self.canvas.axes.barh(speed_data.index, speed_data.values, color='coral')
            self.canvas.axes.set_xlabel('Average Speed (km/h)')
            self.canvas.axes.set_title('Average Speed by Delivery Mode')

        elif chart_type == "Distance Distribution":
            self.canvas.axes.hist(self.df['distance_km'], bins=30, 
                                  edgecolor='black', color='gold')
            self.canvas.axes.axvline(self.df['distance_km'].mean(), 
                                     color='red', linestyle='--',
                                     label=f'Mean: {self.df["distance_km"].mean():.1f}')
            self.canvas.axes.set_xlabel('Distance (km)')
            self.canvas.axes.set_ylabel('Frequency')
            self.canvas.axes.set_title('Distribution of Delivery Distance')
            self.canvas.axes.legend()

        elif chart_type == "Route Efficiency by Mode":
            df_temp = self.df.copy()
            df_temp['efficiency'] = df_temp['distance_km'] / df_temp['route_length_km']
            eff_data = df_temp.groupby('delivery_mode')['efficiency'].mean().sort_values()
            self.canvas.axes.bar(eff_data.index, eff_data.values, color='mediumseagreen')
            self.canvas.axes.set_xlabel('Delivery Mode')
            self.canvas.axes.set_ylabel('Route Efficiency')
            self.canvas.axes.set_title('Route Efficiency by Delivery Mode')
            self.canvas.axes.set_ylim(0, 1)

        elif chart_type == "Correlation Heatmap":
            numeric_cols = ['distance_km', 'delivery_time_min', 'route_length_km']
            corr = self.df[numeric_cols].corr()
            im = self.canvas.axes.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1)
            self.canvas.axes.set_xticks(range(len(corr.columns)))
            self.canvas.axes.set_yticks(range(len(corr.index)))
            self.canvas.axes.set_xticklabels(corr.columns, rotation=45, ha='right')
            self.canvas.axes.set_yticklabels(corr.index)
            self.canvas.axes.set_title('Correlation Heatmap')
            for i in range(len(corr.index)):
                for j in range(len(corr.columns)):
                    self.canvas.axes.text(j, i, f'{corr.values[i, j]:.2f}',
                                         ha='center', va='center')
            self.canvas.fig.colorbar(im, ax=self.canvas.axes)

        elif chart_type == "Orders by Day of Week":
            df_temp = self.df.copy()
            df_temp['day'] = pd.to_datetime(df_temp['order_time']).dt.day_name()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_counts = df_temp['day'].value_counts().reindex(day_order)
            self.canvas.axes.bar(range(7), day_counts.values, color='mediumpurple')
            self.canvas.axes.set_xticks(range(7))
            self.canvas.axes.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
            self.canvas.axes.set_xlabel('Day of Week')
            self.canvas.axes.set_ylabel('Number of Orders')
            self.canvas.axes.set_title('Orders by Day of Week')

        self.canvas.fig.tight_layout()
        self.canvas.draw()


class MLPredictionTab(QWidget):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.init_ui()
        self.train_model()

    def init_ui(self):
        layout = QHBoxLayout()

        # Left panel - Input
        left_panel = QGroupBox("Prediction Input")
        left_layout = QFormLayout()

        self.distance_input = QDoubleSpinBox()
        self.distance_input.setRange(0, 50)
        self.distance_input.setValue(5)
        self.distance_input.setSingleStep(0.5)
        left_layout.addRow("Distance (km):", self.distance_input)

        self.route_length_input = QDoubleSpinBox()
        self.route_length_input.setRange(0, 100)
        self.route_length_input.setValue(7)
        self.route_length_input.setSingleStep(0.5)
        left_layout.addRow("Route Length (km):", self.route_length_input)

        self.hour_input = QSpinBox()
        self.hour_input.setRange(0, 23)
        self.hour_input.setValue(12)
        left_layout.addRow("Hour of Day:", self.hour_input)

        self.traffic_input = QComboBox()
        self.traffic_input.addItems(['Low', 'Medium', 'High'])
        left_layout.addRow("Traffic Level:", self.traffic_input)

        self.mode_input = QComboBox()
        self.mode_input.addItems(self.df['delivery_mode'].unique().tolist())
        left_layout.addRow("Delivery Mode:", self.mode_input)

        self.weather_input = QComboBox()
        self.weather_input.addItems(self.df['weather'].unique().tolist())
        left_layout.addRow("Weather:", self.weather_input)

        self.rest_zone_input = QComboBox()
        self.rest_zone_input.addItems(self.df['restaurant_zone'].unique().tolist())
        left_layout.addRow("Restaurant Zone:", self.rest_zone_input)

        self.cust_zone_input = QComboBox()
        self.cust_zone_input.addItems(self.df['customer_zone'].unique().tolist())
        left_layout.addRow("Customer Zone:", self.cust_zone_input)

        self.predict_btn = QPushButton("Predict Delivery Time")
        self.predict_btn.clicked.connect(self.make_prediction)
        self.predict_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        left_layout.addRow(self.predict_btn)

        left_panel.setLayout(left_layout)
        layout.addWidget(left_panel)

        # Right panel - Results and Model Info
        right_widget = QWidget()
        right_layout = QVBoxLayout()

        # Prediction result
        result_group = QGroupBox("Prediction Result")
        result_layout = QVBoxLayout()
        self.result_label = QLabel("Enter values and click Predict")
        self.result_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("padding: 20px; background-color: #f0f0f0; border-radius: 5px;")
        result_layout.addWidget(self.result_label)
        result_group.setLayout(result_layout)
        right_layout.addWidget(result_group)

        # Model info
        model_group = QGroupBox("Model Information")
        model_layout = QVBoxLayout()
        self.model_info_text = QTextEdit()
        self.model_info_text.setReadOnly(True)
        self.model_info_text.setFont(QFont("Courier", 9))
        model_layout.addWidget(self.model_info_text)
        model_group.setLayout(model_layout)
        right_layout.addWidget(model_group)

        # Retrain button
        self.retrain_btn = QPushButton("Retrain Model")
        self.retrain_btn.clicked.connect(self.train_model)
        right_layout.addWidget(self.retrain_btn)

        right_widget.setLayout(right_layout)
        layout.addWidget(right_widget)

        self.setLayout(layout)

    def train_model(self):
        ml_df = self.df.copy()
        ml_df['hour'] = pd.to_datetime(ml_df['order_time']).dt.hour

        categorical_features = ['traffic_level', 'delivery_mode', 'weather', 
                               'restaurant_zone', 'customer_zone']

        for col in categorical_features:
            le = LabelEncoder()
            ml_df[col + '_encoded'] = le.fit_transform(ml_df[col])
            self.label_encoders[col] = le

        feature_cols = ['distance_km', 'route_length_km', 'hour',
                       'traffic_level_encoded', 'delivery_mode_encoded',
                       'weather_encoded', 'restaurant_zone_encoded', 
                       'customer_zone_encoded']

        X = ml_df[feature_cols].values
        y = ml_df['delivery_time_min'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.model.fit(X_train_scaled, y_train)

        y_pred = self.model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        model_info = "MODEL: Random Forest\n"
        model_info += "=" * 40 + "\n\n"
        model_info += "Performance Metrics:\n"
        model_info += f"  R2 Score: {r2:.4f}\n"
        model_info += f"  RMSE: {rmse:.4f} min\n"
        model_info += f"  MAE: {mae:.4f} min\n\n"
        model_info += "Feature Importance:\n"
        for _, row in importance.iterrows():
            model_info += f"  {row['feature']}: {row['importance']:.4f}\n"

        self.model_info_text.setText(model_info)

    def make_prediction(self):
        try:
            traffic_encoded = self.label_encoders['traffic_level'].transform([self.traffic_input.currentText()])[0]
            mode_encoded = self.label_encoders['delivery_mode'].transform([self.mode_input.currentText()])[0]
            weather_encoded = self.label_encoders['weather'].transform([self.weather_input.currentText()])[0]
            rest_zone_encoded = self.label_encoders['restaurant_zone'].transform([self.rest_zone_input.currentText()])[0]
            cust_zone_encoded = self.label_encoders['customer_zone'].transform([self.cust_zone_input.currentText()])[0]

            features = np.array([[
                self.distance_input.value(),
                self.route_length_input.value(),
                self.hour_input.value(),
                traffic_encoded,
                mode_encoded,
                weather_encoded,
                rest_zone_encoded,
                cust_zone_encoded
            ]])

            features_scaled = self.scaler.transform(features)

            prediction = self.model.predict(features_scaled)[0]

            self.result_label.setText(f"Predicted Delivery Time:\n{prediction:.1f} minutes")
            self.result_label.setStyleSheet("padding: 20px; background-color: #c8e6c9; border-radius: 5px; color: #2e7d32;")

        except Exception as e:
            QMessageBox.warning(self, "Prediction Error", str(e))


class ComparisonTab(QWidget):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Comparison controls
        control_group = QGroupBox("Comparison Settings")
        control_layout = QHBoxLayout()

        control_layout.addWidget(QLabel("Compare by:"))
        self.compare_by_combo = QComboBox()
        self.compare_by_combo.addItems(['delivery_mode', 'traffic_level', 'weather', 
                                        'restaurant_zone', 'customer_zone'])
        control_layout.addWidget(self.compare_by_combo)

        control_layout.addWidget(QLabel("Metric:"))
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(['delivery_time_min', 'distance_km', 'route_length_km'])
        control_layout.addWidget(self.metric_combo)

        self.compare_btn = QPushButton("Generate Comparison")
        self.compare_btn.clicked.connect(self.generate_comparison)
        control_layout.addWidget(self.compare_btn)

        control_layout.addStretch()
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # Splitter for chart and table
        splitter = QSplitter(Qt.Vertical)

        # Chart
        chart_widget = QWidget()
        chart_layout = QVBoxLayout()
        self.canvas = MplCanvas(self, width=10, height=5, dpi=100)
        self.toolbar = NavigationToolbar(self.canvas, self)
        chart_layout.addWidget(self.toolbar)
        chart_layout.addWidget(self.canvas)
        chart_widget.setLayout(chart_layout)
        splitter.addWidget(chart_widget)

        # Comparison table
        table_widget = QWidget()
        table_layout = QVBoxLayout()
        table_layout.addWidget(QLabel("Detailed Comparison:"))
        self.comparison_table = QTableWidget()
        table_layout.addWidget(self.comparison_table)
        table_widget.setLayout(table_layout)
        splitter.addWidget(table_widget)

        layout.addWidget(splitter)
        self.setLayout(layout)

        self.generate_comparison()

    def generate_comparison(self):
        compare_by = self.compare_by_combo.currentText()
        metric = self.metric_combo.currentText()

        # Generate statistics
        grouped = self.df.groupby(compare_by)[metric].agg(['mean', 'std', 'min', 'max', 'count'])
        grouped = grouped.round(2)

        # Update chart
        self.canvas.axes.clear()
        x = range(len(grouped))
        means = grouped['mean'].values
        stds = grouped['std'].values

        bars = self.canvas.axes.bar(x, means, yerr=stds, capsize=5, 
                                    color='steelblue', edgecolor='navy', alpha=0.7)
        self.canvas.axes.set_xticks(x)
        self.canvas.axes.set_xticklabels(grouped.index, rotation=45, ha='right')
        self.canvas.axes.set_xlabel(compare_by)
        self.canvas.axes.set_ylabel(f'{metric} (mean +/- std)')
        self.canvas.axes.set_title(f'{metric} by {compare_by}')
        self.canvas.fig.tight_layout()
        self.canvas.draw()

        # Update table
        self.comparison_table.setColumnCount(len(grouped.columns))
        self.comparison_table.setRowCount(len(grouped))
        self.comparison_table.setHorizontalHeaderLabels(grouped.columns.tolist())
        self.comparison_table.setVerticalHeaderLabels(grouped.index.tolist())

        for i, (idx, row) in enumerate(grouped.iterrows()):
            for j, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.comparison_table.setItem(i, j, item)

        self.comparison_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)


class InsightsTab(QWidget):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout()

        # Key metrics
        metrics_group = QGroupBox("Key Performance Metrics")
        metrics_layout = QGridLayout()

        # Calculate metrics
        avg_delivery_time = self.df['delivery_time_min'].mean()
        avg_distance = self.df['distance_km'].mean()
        avg_speed = (self.df['distance_km'] / self.df['delivery_time_min'] * 60).mean()
        total_orders = len(self.df)
        avg_route_efficiency = (self.df['distance_km'] / self.df['route_length_km']).mean()

        metrics = [
            ("Total Orders", f"{total_orders:,}"),
            ("Avg Delivery Time", f"{avg_delivery_time:.1f} min"),
            ("Avg Distance", f"{avg_distance:.1f} km"),
            ("Avg Speed", f"{avg_speed:.1f} km/h"),
            ("Avg Route Efficiency", f"{avg_route_efficiency:.1%}")
        ]

        for i, (label, value) in enumerate(metrics):
            frame = QFrame()
            frame.setFrameStyle(QFrame.Box | QFrame.Raised)
            frame.setStyleSheet("background-color: #e3f2fd; padding: 10px;")
            frame_layout = QVBoxLayout()
            
            label_widget = QLabel(label)
            label_widget.setAlignment(Qt.AlignCenter)
            frame_layout.addWidget(label_widget)
            
            value_widget = QLabel(value)
            value_widget.setFont(QFont("Arial", 16, QFont.Bold))
            value_widget.setAlignment(Qt.AlignCenter)
            value_widget.setStyleSheet("color: #1565c0;")
            frame_layout.addWidget(value_widget)
            
            frame.setLayout(frame_layout)
            metrics_layout.addWidget(frame, 0, i)

        metrics_group.setLayout(metrics_layout)
        content_layout.addWidget(metrics_group)

        # Insights text
        insights_group = QGroupBox("Data Insights")
        insights_layout = QVBoxLayout()
        
        insights_text = QTextEdit()
        insights_text.setReadOnly(True)
        insights_text.setFont(QFont("Arial", 10))
        
        # Generate insights
        insights = self.generate_insights()
        insights_text.setHtml(insights)
        insights_layout.addWidget(insights_text)
        insights_group.setLayout(insights_layout)
        content_layout.addWidget(insights_group)

        # Recommendations
        recommendations_group = QGroupBox("Recommendations")
        rec_layout = QVBoxLayout()
        rec_text = QTextEdit()
        rec_text.setReadOnly(True)
        rec_text.setFont(QFont("Arial", 10))
        rec_text.setHtml(self.generate_recommendations())
        rec_layout.addWidget(rec_text)
        recommendations_group.setLayout(rec_layout)
        content_layout.addWidget(recommendations_group)

        content.setLayout(content_layout)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        self.setLayout(layout)

    def generate_insights(self):
        insights = "<h3>Key Findings:</h3><ul>"
        
        # Best and worst delivery modes
        mode_times = self.df.groupby('delivery_mode')['delivery_time_min'].mean()
        fastest_mode = mode_times.idxmin()
        slowest_mode = mode_times.idxmax()
        insights += f"<li><b>Fastest delivery mode:</b> {fastest_mode} ({mode_times[fastest_mode]:.1f} min avg)</li>"
        insights += f"<li><b>Slowest delivery mode:</b> {slowest_mode} ({mode_times[slowest_mode]:.1f} min avg)</li>"
        
        # Traffic impact
        traffic_times = self.df.groupby('traffic_level')['delivery_time_min'].mean()
        traffic_diff = traffic_times.get('High', 0) - traffic_times.get('Low', 0)
        insights += f"<li><b>Traffic impact:</b> High traffic adds ~{traffic_diff:.1f} min compared to low traffic</li>"
        
        # Weather impact
        weather_times = self.df.groupby('weather')['delivery_time_min'].mean()
        worst_weather = weather_times.idxmax()
        best_weather = weather_times.idxmin()
        insights += f"<li><b>Best weather for delivery:</b> {best_weather}</li>"
        insights += f"<li><b>Most challenging weather:</b> {worst_weather}</li>"
        
        # Peak hours
        df_temp = self.df.copy()
        df_temp['hour'] = pd.to_datetime(df_temp['order_time']).dt.hour
        hourly = df_temp.groupby('hour')['order_id'].count()
        peak_hour = hourly.idxmax()
        insights += f"<li><b>Peak order hour:</b> {peak_hour}:00 with {hourly[peak_hour]} orders</li>"
        
        # Busiest route
        route_counts = self.df.groupby(['restaurant_zone', 'customer_zone']).size()
        busiest_route = route_counts.idxmax()
        insights += f"<li><b>Busiest route:</b> {busiest_route[0]} to {busiest_route[1]}</li>"
        
        insights += "</ul>"
        return insights

    def generate_recommendations(self):
        rec = "<h3>Optimization Recommendations:</h3><ol>"
        
        # Mode recommendations - Fixed version without apply
        df_temp = self.df.copy()
        df_temp['speed_kmh'] = df_temp['distance_km'] / df_temp['delivery_time_min'] * 60
        mode_speeds = df_temp.groupby('delivery_mode')['speed_kmh'].mean().sort_values(ascending=False)
        rec += f"<li>For speed-critical deliveries, prioritize <b>{mode_speeds.index[0]}</b> which averages {mode_speeds.iloc[0]:.1f} km/h</li>"
        
        # Traffic-based recommendations
        rec += "<li>Implement dynamic routing during high-traffic periods to reduce delays</li>"
        
        # Zone optimization
        zone_times = self.df.groupby(['restaurant_zone', 'customer_zone'])['delivery_time_min'].mean()
        slowest_route = zone_times.idxmax()
        rec += f"<li>Focus optimization efforts on {slowest_route[0]} to {slowest_route[1]} route (avg {zone_times[slowest_route]:.1f} min)</li>"
        
        # Weather preparation
        rec += "<li>Ensure adequate fleet availability during rainy conditions</li>"
        
        # Peak hour management
        rec += "<li>Consider incentivizing off-peak orders to balance demand</li>"
        
        rec += "</ol>"
        return rec


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Food Delivery Route Efficiency Analysis")
        self.setGeometry(100, 100, 1400, 900)
        
        # Load data
        self.df = pd.read_csv('Food_Delivery_Route_Efficiency_Dataset.csv')
        
        self.init_ui()
        self.create_menu()
        self.create_toolbar()
        self.create_statusbar()

    def init_ui(self):
        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Main layout
        layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("Food Delivery Route Efficiency Analysis Dashboard")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #1565c0; padding: 10px;")
        layout.addWidget(title_label)
        
        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setFont(QFont("Arial", 10))
        
        # Add tabs
        self.tabs.addTab(DataExplorerTab(self.df), "Data Explorer")
        self.tabs.addTab(StatisticsTab(self.df), "Statistics")
        self.tabs.addTab(VisualizationTab(self.df), "Visualizations")
        self.tabs.addTab(ComparisonTab(self.df), "Comparison Analysis")
        self.tabs.addTab(MLPredictionTab(self.df), "ML Prediction")
        self.tabs.addTab(InsightsTab(self.df), "Insights")
        
        layout.addWidget(self.tabs)
        main_widget.setLayout(layout)

    def create_menu(self):
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        load_action = QAction("Load Dataset", self)
        load_action.triggered.connect(self.load_dataset)
        file_menu.addAction(load_action)
        
        export_action = QAction("Export Data", self)
        export_action.triggered.connect(self.export_data)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        refresh_action = QAction("Refresh All", self)
        refresh_action.triggered.connect(self.refresh_all)
        view_menu.addAction(refresh_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        
        load_btn = QPushButton("Load Data")
        load_btn.clicked.connect(self.load_dataset)
        toolbar.addWidget(load_btn)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_all)
        toolbar.addWidget(refresh_btn)
        
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self.export_data)
        toolbar.addWidget(export_btn)

    def create_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage(f"Loaded {len(self.df)} records | Ready")

    def load_dataset(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv)")
        if filename:
            try:
                self.df = pd.read_csv(filename)
                self.refresh_all()
                self.statusbar.showMessage(f"Loaded {len(self.df)} records from {filename}")
                QMessageBox.information(self, "Success", "Dataset loaded successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load dataset: {str(e)}")

    def export_data(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
        if filename:
            try:
                self.df.to_csv(filename, index=False)
                QMessageBox.information(self, "Success", f"Data exported to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export: {str(e)}")

    def refresh_all(self):
        # Rebuild tabs with current data
        current_tab = self.tabs.currentIndex()
        
        while self.tabs.count() > 0:
            self.tabs.removeTab(0)
        
        self.tabs.addTab(DataExplorerTab(self.df), "Data Explorer")
        self.tabs.addTab(StatisticsTab(self.df), "Statistics")
        self.tabs.addTab(VisualizationTab(self.df), "Visualizations")
        self.tabs.addTab(ComparisonTab(self.df), "Comparison Analysis")
        self.tabs.addTab(MLPredictionTab(self.df), "ML Prediction")
        self.tabs.addTab(InsightsTab(self.df), "Insights")
        
        self.tabs.setCurrentIndex(current_tab)
        self.statusbar.showMessage(f"Refreshed | {len(self.df)} records")

    def show_about(self):
        about_text = """
        - Data exploration and filtering
        - Statistical analysis
        - Interactive visualizations
        - Comparison analysis
        - Machine learning predictions
        - Business insights and recommendations
        
        Built with PyQT5
        """
        QMessageBox.about(self, "About", about_text)


def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = MainWindow()
    window.showMaximized()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
