import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:fl_chart/fl_chart.dart';

void main() {
  runApp(const PostureMonitorApp());
}

class PostureMonitorApp extends StatelessWidget {
  const PostureMonitorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Posture Monitor',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.deepPurple,
          brightness: Brightness.light,
        ),
        useMaterial3: true,
      ),
      darkTheme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.deepPurple,
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
      ),
      home: const DashboardPage(),
    );
  }
}

class DashboardPage extends StatefulWidget {
  const DashboardPage({super.key});

  @override
  State<DashboardPage> createState() => _DashboardPageState();
}

class _DashboardPageState extends State<DashboardPage> {
  String apiUrl = 'http://10.114.34.209:5000/data'; // Default, user can change
  bool isLoading = false;
  String? errorMessage;
  
  List<LogEntry> logs = [];
  List<SessionEntry> focusSessions = [];
  List<SessionEntry> idleSessions = [];
  
  double avgOverallScore = 0;
  double avgNeckScore = 0;
  double avgTorsoScore = 0;

  @override
  void initState() {
    super.initState();
    // Don't auto-fetch on init, wait for user to set IP
  }

  Future<void> fetchData() async {
    setState(() {
      isLoading = true;
      errorMessage = null;
    });

    try {
      final response = await http.get(Uri.parse(apiUrl));
      
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        
        setState(() {
          // Parse logs
          logs = (data['logs'] as List)
              .map((item) => LogEntry.fromJson(item))
              .toList();
          
          // Parse focus sessions
          focusSessions = (data['focus'] as List)
              .map((item) => SessionEntry.fromJson(item))
              .toList();
          
          // Parse idle sessions
          idleSessions = (data['idle'] as List)
              .map((item) => SessionEntry.fromJson(item))
              .toList();
          
          // Calculate averages
          if (logs.isNotEmpty) {
            avgOverallScore = logs.map((e) => e.overallScore).reduce((a, b) => a + b) / logs.length;
            avgNeckScore = logs.map((e) => e.neckScore).reduce((a, b) => a + b) / logs.length;
            avgTorsoScore = logs.map((e) => e.torsoScore).reduce((a, b) => a + b) / logs.length;
          }
          
          isLoading = false;
        });
      } else {
        setState(() {
          errorMessage = 'Failed to load data: ${response.statusCode}';
          isLoading = false;
        });
      }
    } catch (e) {
      setState(() {
        errorMessage = 'Error: $e';
        isLoading = false;
      });
    }
  }

  void showUrlDialog() {
    final controller = TextEditingController(text: apiUrl);
    
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Set API URL'),
        content: TextField(
          controller: controller,
          decoration: const InputDecoration(
            labelText: 'API URL',
            hintText: 'http://192.168.1.100:5000/data',
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () {
              setState(() {
                apiUrl = controller.text;
              });
              Navigator.pop(context);
              fetchData();
            },
            child: const Text('Connect'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Posture Monitor Dashboard'),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: showUrlDialog,
            tooltip: 'Set API URL',
          ),
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: isLoading ? null : fetchData,
            tooltip: 'Refresh',
          ),
        ],
      ),
      body: isLoading
          ? const Center(child: CircularProgressIndicator())
          : errorMessage != null
              ? Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const Icon(Icons.error_outline, size: 64, color: Colors.red),
                      const SizedBox(height: 16),
                      Text(errorMessage!, textAlign: TextAlign.center),
                      const SizedBox(height: 16),
                      FilledButton.icon(
                        onPressed: showUrlDialog,
                        icon: const Icon(Icons.settings),
                        label: const Text('Configure API'),
                      ),
                    ],
                  ),
                )
              : logs.isEmpty
                  ? Center(
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          const Icon(Icons.analytics_outlined, size: 64),
                          const SizedBox(height: 16),
                          const Text('No data available'),
                          const SizedBox(height: 16),
                          FilledButton.icon(
                            onPressed: showUrlDialog,
                            icon: const Icon(Icons.cloud_download),
                            label: const Text('Fetch Data'),
                          ),
                        ],
                      ),
                    )
                  : _buildDashboard(),
    );
  }

  Widget _buildDashboard() {
    return RefreshIndicator(
      onRefresh: fetchData,
      child: SingleChildScrollView(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Summary Cards
            _buildSummaryCards(),
            const SizedBox(height: 24),
            
            // Posture Score Time Series
            _buildSectionTitle('Posture Scores Over Time'),
            const SizedBox(height: 8),
            _buildScoreTimeSeriesChart(),
            const SizedBox(height: 24),
            
            // Focus Sessions
            if (focusSessions.isNotEmpty) ...[
              _buildSectionTitle('Focus Sessions'),
              const SizedBox(height: 8),
              _buildSessionChart(focusSessions, Colors.cyan),
              const SizedBox(height: 24),
            ],
            
            // Idle Sessions
            if (idleSessions.isNotEmpty) ...[
              _buildSectionTitle('Idle Sessions'),
              const SizedBox(height: 8),
              _buildSessionChart(idleSessions, Colors.orange),
              const SizedBox(height: 24),
            ],
            
            // Recent Logs
            _buildSectionTitle('Recent Activity'),
            const SizedBox(height: 8),
            _buildRecentLogs(),
          ],
        ),
      ),
    );
  }

  Widget _buildSummaryCards() {
    return Row(
      children: [
        Expanded(
          child: _buildSummaryCard(
            'Overall Score',
            avgOverallScore.toStringAsFixed(1),
            Icons.insights,
            _getScoreColor(avgOverallScore),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: _buildSummaryCard(
            'Neck Score',
            avgNeckScore.toStringAsFixed(1),
            Icons.accessibility_new,
            _getScoreColor(avgNeckScore),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: _buildSummaryCard(
            'Torso Score',
            avgTorsoScore.toStringAsFixed(1),
            Icons.accessibility,
            _getScoreColor(avgTorsoScore),
          ),
        ),
      ],
    );
  }

  Widget _buildSummaryCard(String title, String value, IconData icon, Color color) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            Icon(icon, size: 32, color: color),
            const SizedBox(height: 8),
            Text(
              value,
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: color,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              title,
              style: const TextStyle(fontSize: 12),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSectionTitle(String title) {
    return Text(
      title,
      style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
    );
  }

  Widget _buildScoreTimeSeriesChart() {
    if (logs.isEmpty) return const SizedBox();
    
    // Take last 50 entries for readability
    final displayLogs = logs.length > 50 ? logs.sublist(logs.length - 50) : logs;
    
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: SizedBox(
          height: 250,
          child: LineChart(
            LineChartData(
              gridData: FlGridData(show: true, drawVerticalLine: false),
              titlesData: FlTitlesData(
                leftTitles: AxisTitles(
                  sideTitles: SideTitles(
                    showTitles: true,
                    reservedSize: 40,
                    getTitlesWidget: (value, meta) => Text(
                      value.toInt().toString(),
                      style: const TextStyle(fontSize: 10),
                    ),
                  ),
                ),
                bottomTitles: AxisTitles(
                  sideTitles: SideTitles(showTitles: false),
                ),
                topTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
                rightTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
              ),
              borderData: FlBorderData(show: true),
              minY: 0,
              maxY: 100,
              lineBarsData: [
                LineChartBarData(
                  spots: displayLogs.asMap().entries.map((e) {
                    return FlSpot(e.key.toDouble(), e.value.overallScore);
                  }).toList(),
                  isCurved: true,
                  color: Colors.purple,
                  barWidth: 2,
                  dotData: FlDotData(show: false),
                ),
                LineChartBarData(
                  spots: displayLogs.asMap().entries.map((e) {
                    return FlSpot(e.key.toDouble(), e.value.neckScore);
                  }).toList(),
                  isCurved: true,
                  color: Colors.blue,
                  barWidth: 2,
                  dotData: FlDotData(show: false),
                ),
                LineChartBarData(
                  spots: displayLogs.asMap().entries.map((e) {
                    return FlSpot(e.key.toDouble(), e.value.torsoScore);
                  }).toList(),
                  isCurved: true,
                  color: Colors.green,
                  barWidth: 2,
                  dotData: FlDotData(show: false),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildSessionChart(List<SessionEntry> sessions, Color color) {
    if (sessions.isEmpty) return const SizedBox();
    
    // Take last 20 sessions for readability
    final displaySessions = sessions.length > 20 ? sessions.sublist(sessions.length - 20) : sessions;
    
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: SizedBox(
          height: 200,
          child: BarChart(
            BarChartData(
              gridData: FlGridData(show: true, drawVerticalLine: false),
              titlesData: FlTitlesData(
                leftTitles: AxisTitles(
                  sideTitles: SideTitles(
                    showTitles: true,
                    reservedSize: 50,
                    getTitlesWidget: (value, meta) => Text(
                      '${(value / 60).toInt()}m',
                      style: const TextStyle(fontSize: 10),
                    ),
                  ),
                ),
                bottomTitles: AxisTitles(
                  sideTitles: SideTitles(showTitles: false),
                ),
                topTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
                rightTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
              ),
              borderData: FlBorderData(show: true),
              barGroups: displaySessions.asMap().entries.map((e) {
                return BarChartGroupData(
                  x: e.key,
                  barRods: [
                    BarChartRodData(
                      toY: e.value.timePeriod,
                      color: color,
                      width: 8,
                    ),
                  ],
                );
              }).toList(),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildRecentLogs() {
    final recentLogs = logs.length > 10 ? logs.sublist(logs.length - 10).reversed : logs.reversed;
    
    return Card(
      child: ListView.separated(
        shrinkWrap: true,
        physics: const NeverScrollableScrollPhysics(),
        itemCount: recentLogs.length,
        separatorBuilder: (context, index) => const Divider(height: 1),
        itemBuilder: (context, index) {
          final log = recentLogs.elementAt(index);
          return ListTile(
            leading: CircleAvatar(
              backgroundColor: _getScoreColor(log.overallScore),
              child: Text(
                log.overallScore.toInt().toString(),
                style: const TextStyle(color: Colors.white, fontSize: 12),
              ),
            ),
            title: Text(log.timestamp),
            subtitle: Text('Neck: ${log.neckScore.toStringAsFixed(1)} | Torso: ${log.torsoScore.toStringAsFixed(1)}'),
          );
        },
      ),
    );
  }

  Color _getScoreColor(double score) {
    if (score >= 75) return Colors.green;
    if (score >= 50) return Colors.orange;
    return Colors.red;
  }
}

// Data Models
class LogEntry {
  final String timestamp;
  final double overallScore;
  final double neckScore;
  final double torsoScore;

  LogEntry({
    required this.timestamp,
    required this.overallScore,
    required this.neckScore,
    required this.torsoScore,
  });

  factory LogEntry.fromJson(Map<String, dynamic> json) {
    return LogEntry(
      timestamp: json['timestamp'] as String,
      overallScore: double.parse(json['overall_score'].toString()),
      neckScore: double.parse(json['neck_score'].toString()),
      torsoScore: double.parse(json['torso_score'].toString()),
    );
  }
}

class SessionEntry {
  final String startTime;
  final String endTime;
  final double timePeriod;

  SessionEntry({
    required this.startTime,
    required this.endTime,
    required this.timePeriod,
  });

  factory SessionEntry.fromJson(Map<String, dynamic> json) {
    return SessionEntry(
      startTime: json['start_time'] as String,
      endTime: json['end_time'] as String,
      timePeriod: double.parse(json['time_period'].toString()),
    );
  }
}
