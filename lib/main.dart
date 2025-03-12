import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:video_player/video_player.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:io';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: Colors.black,
        primaryColor: Colors.blueAccent,
      ),
      home: VideoUploader(),
    );
  }
}

class VideoUploader extends StatefulWidget {
  @override
  _VideoUploaderState createState() => _VideoUploaderState();
}

class _VideoUploaderState extends State<VideoUploader> {
  File? _video;
  File? _processedVideo;
  VideoPlayerController? _controller;
  final picker = ImagePicker();
  bool _isUploading = false;

  Future<void> pickVideo() async {
    final pickedFile = await picker.pickVideo(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        _video = File(pickedFile.path);
        _processedVideo = null; // Reset processed video on new upload
        _controller?.dispose();
      });
    }
  }

  Future<void> uploadVideo() async {
    if (_video == null) return;

    setState(() {
      _isUploading = true;
    });

    var request = http.MultipartRequest(
      'POST',
      Uri.parse('http://192.168.164.227:5000/upload_video'), // Replace with your server URL
    );
    request.files.add(
      await http.MultipartFile.fromPath('video', _video!.path),
    );

    var response = await request.send();

    if (response.statusCode == 200) {
      // Save the received video file
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
          });
      });

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Upload and processing successful!')),
      );
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Upload failed!')),
      );
    }

    setState(() {
      _isUploading = false;
    });
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Chucking Detection'), centerTitle: true),
      body: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            _video == null
                ? Icon(Icons.video_collection, size: 80, color: Colors.grey)
                : Column(
                    children: [
                      Text(
                        'Selected: ${_video!.path.split('/').last}',
                        style: TextStyle(color: Colors.white70),
                      ),
                    ],
                  ),
            SizedBox(height: 20),
            ElevatedButton.icon(
              onPressed: pickVideo,
              icon: Icon(Icons.video_library),
              label: Text('Pick Video'),
            ),
            SizedBox(height: 10),
            ElevatedButton.icon(
              onPressed: _isUploading ? null : uploadVideo,
              icon: Icon(Icons.upload_file),
              label: _isUploading
                  ? SizedBox(
                      width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : Text('Upload Video'),
            ),
            SizedBox(height: 20),
            _processedVideo != null && _controller != null
                ? Column(
                    children: [
                      Text("Processed Video:", style: TextStyle(fontSize: 18, color: Colors.greenAccent)),
                      AspectRatio(
                        aspectRatio: _controller!.value.aspectRatio,
                        child: VideoPlayer(_controller!),
                      ),
                      SizedBox(height: 10),
                      ElevatedButton.icon(
                        onPressed: () {
                          setState(() {
                            _controller!.value.isPlaying ? _controller!.pause() : _controller!.play();
                          });
                        },
                        icon: Icon(_controller!.value.isPlaying ? Icons.pause : Icons.play_arrow),
                        label: Text(_controller!.value.isPlaying ? "Pause" : "Play"),
                      ),
                    ],
                  )
                : Container(),
          ],
        ),
      ),
    );
  }
}
