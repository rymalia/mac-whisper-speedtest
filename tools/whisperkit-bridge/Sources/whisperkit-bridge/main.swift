import Foundation
import WhisperKit
import ArgumentParser
import AVFoundation

@main
struct WhisperKitBridge: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "whisperkit-bridge",
        abstract: "A bridge to use WhisperKit from Python",
        version: "1.0.0"
    )
    
    @Argument(help: "Path to the input audio file (WAV, MP3, M4A, FLAC)")
    var inputFile: String
    
    @Option(name: .shortAndLong, help: "Output format (json or text)")
    var format: String = "json"
    
    @Option(name: .shortAndLong, help: "Model to use (tiny, base, small, medium, large-v3)")
    var model: String = "base"
    
    func run() async throws {
        // Validate input file exists
        guard FileManager.default.fileExists(atPath: inputFile) else {
            print("Error: Input file does not exist: \(inputFile)", to: &standardError)
            throw ExitCode.failure
        }
        
        do {
            // Initialize WhisperKit with the specified model
            let config = WhisperKitConfig(model: model)
            let whisperKit = try await WhisperKit(config)

            // Transcribe and use WhisperKit's internal timing (excludes audio loading overhead)
            let results = try await whisperKit.transcribe(audioPath: inputFile)

            // Get the pure transcription time from WhisperKit's internal timings
            // This excludes audio loading time and only measures actual transcription processing
            let firstResult = results.first
            let transcriptionTime = firstResult?.timings.fullPipeline ?? 0.0

            // Combine all transcription results
            let combinedText = results.map { $0.text }.joined(separator: " ")

            // Output result
            if format.lowercased() == "json" {
                let output: [String: Any] = [
                    "text": combinedText,
                    "transcription_time": transcriptionTime,  // Pure transcription time (excludes audio loading)
                    "language": firstResult?.language ?? "en",
                    "segments": results.flatMap { result in
                        result.segments.map { segment in
                            [
                                "start": segment.start,
                                "end": segment.end,
                                "text": segment.text
                            ]
                        }
                    }
                ]

                let jsonData = try JSONSerialization.data(withJSONObject: output, options: .prettyPrinted)
                if let jsonString = String(data: jsonData, encoding: .utf8) {
                    print(jsonString)
                }
            } else {
                print(combinedText)
            }
            
        } catch {
            print("Error: \(error)", to: &standardError)
            throw ExitCode.failure
        }
    }
}

var standardError = FileHandle.standardError

extension FileHandle: @retroactive TextOutputStream {
    public func write(_ string: String) {
        let data = Data(string.utf8)
        self.write(data)
    }
}
