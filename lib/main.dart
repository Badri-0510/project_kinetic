import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:video_player/video_player.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:io';
import 'dart:convert';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Cricket Analysis',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        brightness: Brightness.dark,
        scaffoldBackgroundColor: const Color(0xFF0D1117),
        appBarTheme: const AppBarTheme(
          backgroundColor: Color(0xFF161B22),
          elevation: 0,
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: const Color(0xFF2EA043),
            foregroundColor: Colors.white,
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(8),
            ),
          ),
        ),
        cardTheme: CardTheme(
          color: const Color(0xFF161B22),
          elevation: 4,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
        ),
      ),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({Key? key}) : super(key: key);

  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final List<FeatureItem> features = [
    FeatureItem(
      title: 'Chucking Detection',
      description: 'Analyze bowling action for illegal arm straightening',
      icon: Icons.sports_cricket,
      color: Colors.blue,
      screen: ChuckingDetectionScreen(),
    ),
    FeatureItem(
      title: 'No-Ball Detection',
      description: 'Detect front foot no-balls from images',
      icon: Icons.camera_alt,
      color: Colors.red,
      screen: NoBallDetectionScreen(),
    ),
    FeatureItem(
      title: 'Ball Tracking',
      description: 'Track and visualize ball trajectory',
      icon: Icons.track_changes,
      color: Colors.orange,
      screen: BallTrackingScreen(),
    ),
    FeatureItem(
      title: 'Bowling Analysis',
      description: 'Get insights on bowling length and line',
      icon: Icons.analytics,
      color: Colors.green,
      screen: BowlingAnalysisScreen(),
    ),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
    appBar: AppBar(
  title: const Text(
    ' Cricnetic',
    style: TextStyle(
      fontFamily: 'Orbitron',
      fontWeight: FontWeight.bold,
      fontSize: 22,
    ),
  ),
  centerTitle: true,
),
      body: Column(
        children: [
          Container(
            padding: const EdgeInsets.all(16),
            child: const Text(
              'Advanced Cricket Analysis Tools',
              style: TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
          Expanded(
            child: GridView.builder(
              padding: const EdgeInsets.all(16),
              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 2,
                childAspectRatio: 0.6,
                crossAxisSpacing: 16,
                mainAxisSpacing: 16,
              ),
              itemCount: features.length,
              itemBuilder: (context, index) {
                return FeatureCard(feature: features[index]);
              },
            ),
          ),
        ],
      ),
    );
  }
}

class FeatureCard extends StatelessWidget {
  final FeatureItem feature;

  const FeatureCard({Key? key, required this.feature}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () {
        Navigator.push(
          context,
          MaterialPageRoute(builder: (context) => feature.screen),
        );
      },
      child: Card(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                feature.icon,
                size: 48,
                color: feature.color,
              ),
              const SizedBox(height: 16),
              Text(
                feature.title,
                style: const TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 8),
              Text(
                feature.description,
                style: const TextStyle(
                  fontSize: 14,
                  color: Colors.grey,
                ),
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class FeatureItem {
  final String title;
  final String description;
  final IconData icon;
  final Color color;
  final Widget screen;

  FeatureItem({
    required this.title,
    required this.description,
    required this.icon,
    required this.color,
    required this.screen,
  });
}

// Base API URL - CHANGE THIS TO YOUR FLASK SERVER IP
const String baseApiUrl = 'http://192.168.138.227:5000';

// 1. CHUCKING DETECTION SCREEN
class ChuckingDetectionScreen extends StatefulWidget {
  @override
  _ChuckingDetectionScreenState createState() => _ChuckingDetectionScreenState();
}

class _ChuckingDetectionScreenState extends State<ChuckingDetectionScreen> {
  File? _video;
  File? _processedVideo;
  VideoPlayerController? _controller;
  final picker = ImagePicker();
  bool _isUploading = false;
  bool _isPlaying = false;

  Future<void> pickVideo() async {
    final pickedFile = await picker.pickVideo(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        _video = File(pickedFile.path);
        _processedVideo = null;
        _controller?.dispose();
      });
    }
  }

  Future<void> uploadVideo() async {
    if (_video == null) return;

    setState(() {
      _isUploading = true;
    });

    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseApiUrl/upload_video'),
      );
      request.files.add(
        await http.MultipartFile.fromPath('video', _video!.path),
      );

      var response = await request.send();

      if (response.statusCode == 200) {
        final directory = await getApplicationDocumentsDirectory();
        final processedVideoPath = '${directory.path}/processed_video.mp4';
        final processedVideoFile = File(processedVideoPath);
        final bytes = await response.stream.toBytes();
        await processedVideoFile.writeAsBytes(bytes);

        setState(() {
          _processedVideo = processedVideoFile;
          _controller = VideoPlayerController.file(_processedVideo!)
            ..initialize().then((_) {
              setState(() {});
              _controller!.play();
              _isPlaying = true;
            });
        });

        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Analysis completed successfully!')),
        );
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Processing failed! Please try again.')),
        );
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: ${e.toString()}')),
      );
    } finally {
      setState(() {
        _isUploading = false;
      });
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Chucking Detection'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Card(
              child: Padding(
                padding: EdgeInsets.all(16),
                child: Column(
                  children: [
                    Icon(Icons.info_outline, color: Colors.blue, size: 28),
                    SizedBox(height: 8),
                    Text(
                      'This tool analyzes bowling actions to detect illegal straightening of the arm during delivery.',
                      textAlign: TextAlign.center,
                      style: TextStyle(fontSize: 16),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 20),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Upload Video',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 16),
                    Center(
                      child: _video == null
                          ? Container(
                              height: 180,
                              decoration: BoxDecoration(
                                color: Colors.black12,
                                borderRadius: BorderRadius.circular(8),
                              ),
                              child: const Center(
                                child: Column(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: [
                                    Icon(Icons.video_collection, size: 60, color: Colors.grey),
                                    SizedBox(height: 8),
                                    Text('No video selected'),
                                  ],
                                ),
                              ),
                            )
                          : Text(
                              'Selected: ${_video!.path.split('/').last}',
                              style: TextStyle(color: Colors.white70),
                            ),
                    ),
                    const SizedBox(height: 16),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                      children: [
                        ElevatedButton.icon(
                          onPressed: pickVideo,
                          icon: const Icon(Icons.video_library),
                          label: const Text('Pick Video'),
                        ),
                        ElevatedButton.icon(
                          onPressed: _video == null || _isUploading ? null : uploadVideo,
                          icon: _isUploading
                              ? const SizedBox(
                                  width: 20,
                                  height: 20,
                                  child: CircularProgressIndicator(
                                    strokeWidth: 2,
                                    color: Colors.white,
                                  ),
                                )
                              : const Icon(Icons.upload_file),
                          label: Text(_isUploading ? 'Processing...' : 'Analyze'),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 20),
            if (_processedVideo != null && _controller != null && _controller!.value.isInitialized)
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'Analysis Results',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 16),
                      AspectRatio(
                        aspectRatio: _controller!.value.aspectRatio,
                        child: VideoPlayer(_controller!),
                      ),
                      const SizedBox(height: 16),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          IconButton(
                            icon: Icon(
                              _isPlaying ? Icons.pause : Icons.play_arrow,
                              color: Colors.white,
                              size: 30,
                            ),
                            onPressed: () {
                              setState(() {
                                if (_isPlaying) {
                                  _controller!.pause();
                                } else {
                                  _controller!.play();
                                }
                                _isPlaying = !_isPlaying;
                              });
                            },
                          ),
                          IconButton(
                            icon: const Icon(
                              Icons.replay,
                              color: Colors.white,
                              size: 30,
                            ),
                            onPressed: () {
                              _controller!.seekTo(Duration.zero);
                              _controller!.play();
                              setState(() {
                                _isPlaying = true;
                              });
                            },
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

// 2. NO-BALL DETECTION SCREEN
class NoBallDetectionScreen extends StatefulWidget {
  @override
  _NoBallDetectionScreenState createState() => _NoBallDetectionScreenState();
}

class _NoBallDetectionScreenState extends State<NoBallDetectionScreen> {
  File? _image;
  final picker = ImagePicker();
  bool _isUploading = false;
  String? _result;
  double? _confidence;

  Future<void> pickImage(ImageSource source) async {
    final pickedFile = await picker.pickImage(source: source);
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
        _result = null;
        _confidence = null;
      });
    }
  }

  Future<void> uploadImage() async {
    if (_image == null) return;

    setState(() {
      _isUploading = true;
    });

    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseApiUrl/predict_noball'),
      );
      request.files.add(
        await http.MultipartFile.fromPath('image', _image!.path),
      );

      var response = await request.send();
      var responseData = await response.stream.bytesToString();
      var jsonData = jsonDecode(responseData);

      if (response.statusCode == 200) {
        setState(() {
          _result = jsonData['prediction'];
          _confidence = jsonData['confidence'];
        });

        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Analysis completed successfully!')),
        );
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error: ${jsonData['error'] ?? 'Unknown error'}')),
        );
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: ${e.toString()}')),
      );
    } finally {
      setState(() {
        _isUploading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('No-Ball Detection'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Card(
              child: Padding(
                padding: EdgeInsets.all(16),
                child: Column(
                  children: [
                    Icon(Icons.info_outline, color: Colors.red, size: 28),
                    SizedBox(height: 8),
                    Text(
                      'Upload an image of a bowling delivery to detect if it\'s a no-ball.',
                      textAlign: TextAlign.center,
                      style: TextStyle(fontSize: 16),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 20),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Upload Image',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 16),
                    Center(
                      child: _image == null
                          ? Container(
                              height: 200,
                              decoration: BoxDecoration(
                                color: Colors.black12,
                                borderRadius: BorderRadius.circular(8),
                              ),
                              child: const Center(
                                child: Column(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: [
                                    Icon(Icons.image, size: 60, color: Colors.grey),
                                    SizedBox(height: 8),
                                    Text('No image selected'),
                                  ],
                                ),
                              ),
                            )
                          : ClipRRect(
                              borderRadius: BorderRadius.circular(8),
                              child: Image.file(
                                _image!,
                                height: 250,
                                fit: BoxFit.cover,
                              ),
                            ),
                    ),
                    const SizedBox(height: 16),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                      children: [
                        ElevatedButton.icon(
                          onPressed: () => pickImage(ImageSource.gallery),
                          icon: const Icon(Icons.photo_library),
                          label: const Text('Pick Image'),
                        ),
                      
                        ElevatedButton.icon(
                          onPressed: _image == null || _isUploading ? null : uploadImage,
                          icon: _isUploading
                              ? const SizedBox(
                                  width: 20,
                                  height: 20,
                                  child: CircularProgressIndicator(
                                    strokeWidth: 2,
                                    color: Colors.white,
                                  ),
                                )
                              : const Icon(Icons.check_circle),
                          label: Text(_isUploading ? 'Analyzing...' : 'Analyze'),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 20),
            if (_result != null)
              Card(
                color: _result == 'No-Ball' ? Colors.red.shade900 : Colors.green.shade900,
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    children: [
                      Text(
                        _result!,
                        style: const TextStyle(
                          fontSize: 24,
                          fontWeight: FontWeight.bold,
                          color: Colors.white,
                        ),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'Confidence: ${(_confidence! * 100).toStringAsFixed(2)}%',
                        style: const TextStyle(
                          fontSize: 16,
                          color: Colors.white70,
                        ),
                      ),
                      const SizedBox(height: 16),
                      Icon(
                        _result == 'No-Ball' ? Icons.cancel : Icons.check_circle,
                        size: 48,
                        color: Colors.white,
                      ),
                    ],
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

// 3. BALL TRACKING SCREEN
class BallTrackingScreen extends StatefulWidget {
  @override
  _BallTrackingScreenState createState() => _BallTrackingScreenState();
}

class _BallTrackingScreenState extends State<BallTrackingScreen> {
  File? _video;
  File? _processedVideo;
  VideoPlayerController? _controller;
  final picker = ImagePicker();
  bool _isUploading = false;
  bool _isPlaying = false;

  Future<void> pickVideo() async {
    final pickedFile = await picker.pickVideo(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        _video = File(pickedFile.path);
        _processedVideo = null;
        _controller?.dispose();
      });
    }
  }

  Future<void> uploadVideo() async {
    if (_video == null) return;

    setState(() {
      _isUploading = true;
    });

    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseApiUrl/detect_ball_trajectory'),
      );
      request.files.add(
        await http.MultipartFile.fromPath('video', _video!.path),
      );

      var response = await request.send();

      if (response.statusCode == 200) {
        final directory = await getApplicationDocumentsDirectory();
        final processedVideoPath = '${directory.path}/trajectory_video.mp4';
        final processedVideoFile = File(processedVideoPath);
        final bytes = await response.stream.toBytes();
        await processedVideoFile.writeAsBytes(bytes);

        setState(() {
          _processedVideo = processedVideoFile;
          _controller = VideoPlayerController.file(_processedVideo!)
            ..initialize().then((_) {
              setState(() {});
              _controller!.play();
              _isPlaying = true;
            });
        });

        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Trajectory analysis completed!')),
        );
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Processing failed! Please try again.')),
        );
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: ${e.toString()}')),
      );
    } finally {
      setState(() {
        _isUploading = false;
      });
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Ball Trajectory Tracking'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Card(
              child: Padding(
                padding: EdgeInsets.all(16),
                child: Column(
                  children: [
                    Icon(Icons.info_outline, color: Colors.orange, size: 28),
                    SizedBox(height: 8),
                    Text(
                      'This tool tracks and visualizes the ball trajectory in 3D space.',
                      textAlign: TextAlign.center,
                      style: TextStyle(fontSize: 16),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 20),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Upload Video',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 16),
                    Center(
                      child: _video == null
                          ? Container(
                              height: 180,
                              decoration: BoxDecoration(
                                color: Colors.black12,
                                borderRadius: BorderRadius.circular(8),
                              ),
                              child: const Center(
                                child: Column(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: [
                                    Icon(Icons.video_collection, size: 60, color: Colors.grey),
                                    SizedBox(height: 8),
                                    Text('No video selected'),
                                  ],
                                ),
                              ),
                            )
                          : Text(
                              'Selected: ${_video!.path.split('/').last}',
                              style: TextStyle(color: Colors.white70),
                            ),
                    ),
                    const SizedBox(height: 16),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                      children: [
                        ElevatedButton.icon(
                          onPressed: pickVideo,
                          icon: const Icon(Icons.video_library),
                          label: const Text('Pick Video'),
                        ),
                        ElevatedButton.icon(
                          onPressed: _video == null || _isUploading ? null : uploadVideo,
                          icon: _isUploading
                              ? const SizedBox(
                                  width: 20,
                                  height: 20,
                                  child: CircularProgressIndicator(
                                    strokeWidth: 2,
                                    color: Colors.white,
                                  ),
                                )
                              : const Icon(Icons.track_changes),
                          label: Text(_isUploading ? 'Processing...' : 'Track Ball'),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 20),
            if (_processedVideo != null && _controller != null && _controller!.value.isInitialized)
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'Trajectory Analysis',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 16),
                      AspectRatio(
                        aspectRatio: _controller!.value.aspectRatio,
                        child: VideoPlayer(_controller!),
                      ),
                      const SizedBox(height: 16),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          IconButton(
                            icon: Icon(
                              _isPlaying ? Icons.pause : Icons.play_arrow,
                              color: Colors.white,
                              size: 30,
                            ),
                            onPressed: () {
                              setState(() {
                                if (_isPlaying) {
                                  _controller!.pause();
                                } else {
                                  _controller!.play();
                                }
                                _isPlaying = !_isPlaying;
                              });
                            },
                          ),
                          IconButton(
                            icon: const Icon(
                              Icons.replay,
                              color: Colors.white,
                              size: 30,
                            ),
                            onPressed: () {
                              _controller!.seekTo(Duration.zero);
                              _controller!.play();
                              setState(() {
                                _isPlaying = true;
                              });
                            },
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

// 4. BOWLING ANALYSIS SCREEN
class BowlingAnalysisScreen extends StatefulWidget {
  @override
  _BowlingAnalysisScreenState createState() => _BowlingAnalysisScreenState();
}

class _BowlingAnalysisScreenState extends State<BowlingAnalysisScreen> {
  File? _video;
  File? _resultImage;
  final picker = ImagePicker();
  bool _isUploading = false;

  Future<void> pickVideo() async {
    final pickedFile = await picker.pickVideo(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        _video = File(pickedFile.path);
        _resultImage = null;
      });
    }
  }

  Future<void> uploadVideo() async {
    if (_video == null) return;

    setState(() {
      _isUploading = true;
    });

    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseApiUrl/analyze_bowling'),
      );
      request.files.add(
        await http.MultipartFile.fromPath('video', _video!.path),
      );

      var response = await request.send();

      if (response.statusCode == 200) {
        final directory = await getApplicationDocumentsDirectory();
        final imagePath = '${directory.path}/bowling_analysis.jpg';
        final imageFile = File(imagePath);
        final bytes = await response.stream.toBytes();
        await imageFile.writeAsBytes(bytes);

        setState(() {
          _resultImage = imageFile;
        });

        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Bowling analysis completed!')),
        );
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Analysis failed! Please try again.')),
        );
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: ${e.toString()}')),
      );
    } finally {
      setState(() {
        _isUploading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Bowling Analysis'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Card(
              child: Padding(
                padding: EdgeInsets.all(16),
                child: Column(
                  children: [
                    Icon(Icons.info_outline, color: Colors.green, size: 28),
                    SizedBox(height: 8),Text(
                      'Analyze bowling videos to get insights on length, line, and bounce point.',
                      textAlign: TextAlign.center,
                      style: TextStyle(fontSize: 16),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 20),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Upload Video',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 16),
                    Center(
                      child: _video == null
                          ? Container(
                              height: 180,
                              decoration: BoxDecoration(
                                color: Colors.black12,
                                borderRadius: BorderRadius.circular(8),
                              ),
                              child: const Center(
                                child: Column(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: [
                                    Icon(Icons.video_collection, size: 60, color: Colors.grey),
                                    SizedBox(height: 8),
                                    Text('No video selected'),
                                  ],
                                ),
                              ),
                            )
                          : Text(
                              'Selected: ${_video!.path.split('/').last}',
                              style: TextStyle(color: Colors.white70),
                            ),
                    ),
                    const SizedBox(height: 16),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                      children: [
                        ElevatedButton.icon(
                          onPressed: pickVideo,
                          icon: const Icon(Icons.video_library),
                          label: const Text('Pick Video'),
                        ),
                        ElevatedButton.icon(
                          onPressed: _video == null || _isUploading ? null : uploadVideo,
                          icon: _isUploading
                              ? const SizedBox(
                                  width: 20,
                                  height: 20,
                                  child: CircularProgressIndicator(
                                    strokeWidth: 2,
                                    color: Colors.white,
                                  ),
                                )
                              : const Icon(Icons.analytics),
                          label: Text(_isUploading ? 'Analyzing...' : 'Analyze'),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 20),
            if (_resultImage != null)
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'Analysis Results',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 16),
                      ClipRRect(
                        borderRadius: BorderRadius.circular(8),
                        child: Image.file(
                          _resultImage!,
                          width: double.infinity,
                          fit: BoxFit.cover,
                        ),
                      ),
                      const SizedBox(height: 16),
                      const Center(
                        child: Text(
                          'The highlighted areas show pitch regions. Bounce point is marked in red.',
                          textAlign: TextAlign.center,
                          style: TextStyle(
                            fontSize: 14,
                            color: Colors.grey,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}