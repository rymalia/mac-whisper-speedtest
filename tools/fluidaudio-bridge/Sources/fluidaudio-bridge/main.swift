import Foundation
import FluidAudio
import ArgumentParser
import AVFoundation

@main
struct FluidAudioBridge: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "fluidaudio-bridge",
        abstract: "A bridge to use FluidAudio from Python",
        version: "1.0.0"
    )
    
    @Argument(help: "Path to the input audio file (WAV format, 16kHz mono)")
    var inputFile: String
    
    @Option(name: .shortAndLong, help: "Output format (json or text)")
    var format: String = "json"
    
    func run() async throws {
        // Validate input file exists
        let inputURL = URL(fileURLWithPath: inputFile)
        guard FileManager.default.fileExists(atPath: inputFile) else {
            print("Error: Input file does not exist: \(inputFile)", to: &standardError)
            throw ExitCode.failure
        }
        
        do {
            // Load audio file
            let audioData = try loadAudioFile(url: inputURL)

            // Initialize ASR with default config
            // Note: ASRConfig API changed significantly in FluidAudio 0.10.0
            // Old API had: maxSymbolsPerFrame, realtimeMode, chunkSizeMs, tdtConfig
            // New API has: sampleRate, tdtConfig (with different TdtConfig params)
            // Using defaults for compatibility
            let asrConfig = ASRConfig.default

            let asrManager = AsrManager(config: asrConfig)

            // Load models - explicitly specify v2 to maintain compatibility
            // Note: FluidAudio 0.4.0+ defaults to v3, we want v2 for consistency
            let models = try await AsrModels.downloadAndLoad(version: .v2)
            try await asrManager.initialize(models: models)
            
            // Transcribe (measure only the actual transcription time)
            let transcriptionStartTime = CFAbsoluteTimeGetCurrent()
            let result = try await asrManager.transcribe(audioData)
            let transcriptionTime = CFAbsoluteTimeGetCurrent() - transcriptionStartTime

            // Output result
            if format.lowercased() == "json" {
                let output: [String: Any] = [
                    "text": result.text,
                    "transcription_time": transcriptionTime,  // Pure transcription time
                    "processing_time": result.processingTime, // FluidAudio's internal processing time
                    "language": "en"
                ]
                
                let jsonData = try JSONSerialization.data(withJSONObject: output, options: .prettyPrinted)
                if let jsonString = String(data: jsonData, encoding: .utf8) {
                    print(jsonString)
                }
            } else {
                print(result.text)
            }
            
        } catch {
            print("Error: \(error)", to: &standardError)
            throw ExitCode.failure
        }
    }
    
    private func loadAudioFile(url: URL) throws -> [Float] {
        let audioFile = try AVAudioFile(forReading: url)
        let format = audioFile.processingFormat
        
        // Ensure we have the right format (16kHz mono)
        guard format.sampleRate == 16000 && format.channelCount == 1 else {
            throw ValidationError("Audio file must be 16kHz mono. Got \(format.sampleRate)Hz, \(format.channelCount) channels")
        }
        
        let frameCount = UInt32(audioFile.length)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw ValidationError("Failed to create audio buffer")
        }
        
        try audioFile.read(into: buffer)
        
        guard let floatChannelData = buffer.floatChannelData else {
            throw ValidationError("Failed to get float channel data")
        }
        
        let audioData = Array(UnsafeBufferPointer(start: floatChannelData[0], count: Int(buffer.frameLength)))
        return audioData
    }
}

var standardError = FileHandle.standardError

extension FileHandle: @retroactive TextOutputStream {
    public func write(_ string: String) {
        let data = Data(string.utf8)
        self.write(data)
    }
}
