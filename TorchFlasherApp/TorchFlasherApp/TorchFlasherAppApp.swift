import SwiftUI
import AVFoundation
import Swifter
import AudioToolbox


@main
struct TorchFlasherApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

class FlashModel: ObservableObject {
    @Published var lastFlashTime: Date?
    @Published var permissionGranted = false
}

/// Delegate to receive the photo data and log debug info
class PhotoDelegate: NSObject, AVCapturePhotoCaptureDelegate {
    private let cb: (Data?)->Void
    init(_ cb: @escaping (Data?)->Void) {
        self.cb = cb
    }
    func photoOutput(_ output: AVCapturePhotoOutput,
                      willCapturePhotoFor resolvedSettings: AVCaptureResolvedPhotoSettings) {
         // Play the native shutter sound:
         AudioServicesPlaySystemSound(1108)
    }

    func photoOutput(
        _ output: AVCapturePhotoOutput,
        didFinishProcessingPhoto photo: AVCapturePhoto,
        error: Error?
    ) {
        if let e = error {
            print("❌ Photo error: \(e)")
            cb(nil)
        } else if let data = photo.fileDataRepresentation() {
            print("[+] Captured JPEG (\(data.count) bytes)")
            cb(data)
        } else {
            print("❌ fileDataRepresentation() returned nil")
            cb(nil)
        }
    }
}

struct ContentView: View {
    @StateObject private var model = FlashModel()
    @State private var server: HttpServer? = nil
    @State private var pendingPhotoDelegates: [PhotoDelegate] = []

    private let session = AVCaptureSession()
    private let photoOutput = AVCapturePhotoOutput()
    private let port: in_port_t = 8395

    var body: some View {
        VStack(spacing: 20) {
            if model.permissionGranted {
                Text("📸 Flash HTTP Server").font(.title)
                Text("• GET /flash → blinks torch\n• GET /capture → flash photo & return JPEG")
                    .multilineTextAlignment(.center)
                Text("Server on port \(port)").font(.subheadline)
                if let t = model.lastFlashTime {
                    Text("Last at \(t.formatted(.dateTime.hour().minute().second()))")
                        .foregroundColor(.green)
                }
            } else {
                Text("Requesting camera permission…")
            }
        }
        .padding()
        .onAppear { initialize() }
    }

    private func initialize() {
        AVCaptureDevice.requestAccess(for: .video) { granted in
            DispatchQueue.main.async {
                self.model.permissionGranted = granted
                guard granted else {
                    print("[!] Camera access denied")
                    return
                }
                print("[+] Camera access granted")
                configureSession()
                blinkTorch(times: 3, on: 0.3, off: 0.2)
                startServer()
            }
        }
    }

    private func configureSession() {
        DispatchQueue.global(qos: .userInitiated).async {
            let discovery = AVCaptureDevice.DiscoverySession(
                deviceTypes: [.builtInWideAngleCamera,
                              .builtInDualCamera,
                              .builtInTripleCamera,
                              .builtInUltraWideCamera,
                              .builtInTelephotoCamera],
                mediaType: .video,
                position: .back)
            guard let device = discovery.devices.first else {
                print("[!] No back-facing camera found")
                return
            }
            do {
                let input = try AVCaptureDeviceInput(device: device)
                session.beginConfiguration()
                if session.canAddInput(input) { session.addInput(input) }
                if session.canAddOutput(photoOutput) { session.addOutput(photoOutput) }
                session.commitConfiguration()
                session.startRunning()
                print("[+] Capture session started with \(device.localizedName)")
            } catch {
                print("[!] Session configuration failed: \(error)")
            }
        }
    }

    private func startServer() {
        let http = HttpServer()

        http["/flash"] = { _ in
            print("🟢 /flash hit")
            DispatchQueue.main.async {
                blinkTorch(times: 1, on: 0.15, off: 0)
                model.lastFlashTime = Date()
            }
            return .ok(.text("blinked"))
        }

        http["/capture"] = { _ in
            let group = DispatchGroup()
            group.enter()
            var jpeg: Data?
            var delegateRef: PhotoDelegate? = nil

            let settings = AVCapturePhotoSettings(format: [
                AVVideoCodecKey: AVVideoCodecType.jpeg
            ])
            settings.flashMode = .on

            DispatchQueue.main.async {
                let delegate = PhotoDelegate { data in
                    jpeg = data
                    group.leave()
                }
                pendingPhotoDelegates.append(delegate)
                photoOutput.capturePhoto(with: settings, delegate: delegate)
            }

            if group.wait(timeout: .now() + 5) == .success,
               let data = jpeg
            {
                DispatchQueue.main.async {
                    model.lastFlashTime = Date()
                    pendingPhotoDelegates.removeAll { $0 === delegateRef }
                }
                return HttpResponse.raw(
                    200, "OK",
                    ["Content-Type": "image/jpeg"]
                ) { writer in
                    try? writer.write(data)
                }
            } else {
                print("[!] Photo capture timed out or failed")
                return .internalServerError
            }
        }

        do {
            try http.start(port, forceIPv4: true)
            server = http
            print("[+] HTTP server listening on port \(port)")
        } catch {
            print("[!] HTTP server failed: \(error)")
        }
    }

    private func blinkTorch(times: Int, on: TimeInterval, off: TimeInterval) {
        for i in 0..<times {
            DispatchQueue.main.asyncAfter(deadline: .now() + Double(i) * (on + off)) {
                flashTorch(duration: on)
            }
        }
    }
}

func flashTorch(duration: TimeInterval) {
    guard let dev = AVCaptureDevice.default(for: .video), dev.hasTorch else {
        print("[!] Torch unavailable")
        return
    }
    do {
        try dev.lockForConfiguration()
        dev.torchMode = .on
        dev.unlockForConfiguration()
        print("[+] Torch ON")
    } catch {
        print("[!] Error enabling torch: \(error)")
        return
    }
    DispatchQueue.global().asyncAfter(deadline: .now() + duration) {
        do {
            try dev.lockForConfiguration()
            dev.torchMode = .off
            dev.unlockForConfiguration()
            print("[+] Torch OFF")
        } catch {
            print("[!] Error disabling torch: \(error)")
        }
    }
}
