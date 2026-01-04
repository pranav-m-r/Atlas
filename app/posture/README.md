# Posture Monitor Dashboard App

A Flutter mobile app for visualizing posture monitoring data from the Raspberry Pi posture detection system.

## Features

- 📊 **Real-time Dashboard**: View posture metrics and session data
- 📈 **Time Series Charts**: Track posture scores (overall, neck, torso) over time
- ⏱️ **Focus Sessions**: Visualize focus session durations with bar charts
- 🪑 **Idle Sessions**: Monitor idle/sitting session patterns
- 🔄 **Pull to Refresh**: Easy data updates
- ⚙️ **Configurable API**: Set your Raspberry Pi IP address

## Setup

### 1. Install Dependencies

```bash
flutter pub get
```

### 2. Configure Raspberry Pi API

When you first launch the app, tap the **Settings** icon (⚙️) in the app bar and enter your Raspberry Pi's IP address:

```
http://192.168.1.XXX:5000/data
```

Replace `192.168.1.XXX` with your actual Raspberry Pi IP address (shown in the terminal when running `main.py`).

### 3. Run the App

For Android:
```bash
flutter run
```

For Android release build:
```bash
flutter build apk
```

The APK will be generated at: `build/app/outputs/flutter-apk/app-release.apk`

## Usage

1. **Set API URL**: Tap the settings icon and enter your Raspberry Pi's API endpoint
2. **Fetch Data**: Tap "Fetch Data" or use the refresh button
3. **View Analytics**:
   - Summary cards show average scores
   - Line charts display posture score trends
   - Bar charts show focus and idle session durations
   - Recent activity log shows latest posture data

4. **Refresh**: Pull down on the screen or tap the refresh icon to update data

## Dashboard Components

### Summary Cards
- **Overall Score**: Average posture score (color-coded: green ≥75, orange ≥50, red <50)
- **Neck Score**: Average neck angle score
- **Torso Score**: Average torso angle score

### Charts
- **Posture Scores Over Time**: Line chart showing overall, neck, and torso scores
- **Focus Sessions**: Bar chart of focus session durations (in minutes)
- **Idle Sessions**: Bar chart of idle/sitting session durations (in minutes)
- **Recent Activity**: List of latest posture measurements

## Requirements

- Flutter SDK 3.7.2 or higher
- Android device or emulator
- Running Raspberry Pi posture monitor with Flask API

## Dependencies

- `http: ^1.2.0` - HTTP requests to Raspberry Pi API
- `fl_chart: ^0.69.0` - Beautiful charts and graphs

## Troubleshooting

### Cannot connect to API
- Ensure your phone and Raspberry Pi are on the same network
- Verify the IP address in settings matches the Raspberry Pi IP
- Check that `main.py` is running on the Raspberry Pi
- Ensure port 5000 is not blocked by firewall

### No data showing
- Make sure the posture monitor has been running and generating data
- Check that CSV files exist in `src/outputs/` on the Raspberry Pi
- Try refreshing the data

### Charts not displaying
- Ensure there is sufficient data (at least a few log entries)
- Try scrolling down to see all chart sections
