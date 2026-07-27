package com.yolomnn.demo

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.yolomnn.YoloDetector
import com.yolomnn.demo.databinding.ActivityMainBinding
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var detector: YoloDetector
    private val inferExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    private var variant = ModelVariant.N
    private var cameraMode = false

    @Volatile private var confThreshold = 0.25f
    @Volatile private var iouThreshold = 0.45f
    private var lastStillBitmap: Bitmap? = null

    // reused camera state
    private var cameraProvider: ProcessCameraProvider? = null
    @Volatile private var analyzing = false
    private var frameCount = 0
    private var lastFpsTs = 0L

    private val pickImage = registerForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri: Uri? -> uri?.let { runOnStillImage(it) } }

    private val requestCamera = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> if (granted) startCamera() else toast("需要相机权限") }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        detector = YoloDetector()
        loadVariant(variant)

        binding.modelToggle.check(R.id.btnModelN)
        binding.modelToggle.addOnButtonCheckedListener { _, checkedId, isChecked ->
            if (!isChecked) return@addOnButtonCheckedListener
            variant = if (checkedId == R.id.btnModelS) ModelVariant.S else ModelVariant.N
            loadVariant(variant)
        }

        binding.btnPick.setOnClickListener {
            switchToStillMode()
            pickImage.launch(
                androidx.activity.result.PickVisualMediaRequest(
                    ActivityResultContracts.PickVisualMedia.ImageOnly
                )
            )
        }

        binding.btnCamera.setOnClickListener {
            if (cameraMode) switchToStillMode() else enterCameraMode()
        }

        binding.confSlider.value = confThreshold
        binding.iouSlider.value = iouThreshold
        updateThresholdLabel()
        binding.confSlider.addOnChangeListener { _, value, _ ->
            confThreshold = value
            updateThresholdLabel()
            rerunStillIfIdle()
        }
        binding.iouSlider.addOnChangeListener { _, value, _ ->
            iouThreshold = value
            updateThresholdLabel()
            rerunStillIfIdle()
        }
    }

    private fun updateThresholdLabel() {
        binding.thresholdLabel.text = getString(
            R.string.threshold_label, confThreshold, iouThreshold
        )
    }

    /** In still mode, re-run detection on the last image when a slider changes. */
    private fun rerunStillIfIdle() {
        if (cameraMode) return
        val bmp = lastStillBitmap ?: return
        inferExecutor.execute {
            try {
                val result = detector.detect(bmp, confThreshold, iouThreshold)
                runOnUiThread {
                    binding.overlay.setResults(result.detections, bmp.width, bmp.height, fitCenter = true)
                    binding.stats.text = getString(
                        R.string.stats_still, variant.displayName, result.inferenceMs, result.detections.size
                    )
                }
            } catch (e: Exception) {
                Log.e(TAG, "rerun failed", e)
            }
        }
    }

    private fun loadVariant(v: ModelVariant) {
        inferExecutor.execute {
            val path = ModelAssets.ensureExtracted(this, v)
            detector.loadModel(path)
        }
    }

    // ---- still-image (album) mode ----

    private fun switchToStillMode() {
        cameraMode = false
        cameraProvider?.unbindAll()
        binding.previewView.visibility = View.GONE
        binding.imageView.visibility = View.VISIBLE
        binding.btnCamera.text = getString(R.string.camera_start)
        binding.overlay.clear()
    }

    private fun runOnStillImage(uri: Uri) {
        val bitmap = contentResolver.openInputStream(uri).use { input ->
            BitmapFactory.decodeStream(input)
        }?.let { toArgb(it) } ?: run { toast("无法读取图片"); return }

        binding.imageView.setImageBitmap(bitmap)
        binding.overlay.clear()
        lastStillBitmap = bitmap
        inferExecutor.execute {
            try {
                val result = detector.detect(bitmap, confThreshold, iouThreshold)
                runOnUiThread {
                    binding.overlay.setResults(result.detections, bitmap.width, bitmap.height, fitCenter = true)
                    binding.stats.text = getString(
                        R.string.stats_still, variant.displayName, result.inferenceMs, result.detections.size
                    )
                }
            } catch (e: Exception) {
                Log.e(TAG, "still inference failed", e)
                runOnUiThread { toast("推理失败: ${e.message}") }
            }
        }
    }

    // ---- camera (live) mode ----

    private fun enterCameraMode() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            requestCamera.launch(Manifest.permission.CAMERA)
            return
        }
        startCamera()
    }

    private fun startCamera() {
        cameraMode = true
        binding.imageView.visibility = View.GONE
        binding.previewView.visibility = View.VISIBLE
        binding.btnCamera.text = getString(R.string.camera_stop)

        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            cameraProvider = provider

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.previewView.surfaceProvider)
            }
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { it.setAnalyzer(inferExecutor, ::analyzeFrame) }

            try {
                provider.unbindAll()
                provider.bindToLifecycle(
                    this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis
                )
            } catch (e: Exception) {
                Log.e(TAG, "camera bind failed", e)
                toast("相机启动失败")
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyzeFrame(image: ImageProxy) {
        if (analyzing || !cameraMode) { image.close(); return }
        analyzing = true
        try {
            val bitmap = ImageUtils.imageProxyToBitmap(image)
            val result = detector.detect(bitmap, confThreshold, iouThreshold)

            frameCount++
            val now = System.currentTimeMillis()
            if (lastFpsTs == 0L) lastFpsTs = now
            val elapsed = now - lastFpsTs
            val fps = if (elapsed > 0) frameCount * 1000f / elapsed else 0f
            if (elapsed > 1000) { frameCount = 0; lastFpsTs = now }

            val w = bitmap.width
            val h = bitmap.height
            runOnUiThread {
                if (cameraMode) {
                    binding.overlay.setResults(result.detections, w, h, fitCenter = false)
                    binding.stats.text = getString(
                        R.string.stats_live, variant.displayName, result.inferenceMs, fps, result.detections.size
                    )
                }
            }
            bitmap.recycle()
        } catch (e: Exception) {
            Log.e(TAG, "frame analyze failed", e)
        } finally {
            analyzing = false
            image.close()
        }
    }

    private fun toArgb(src: Bitmap): Bitmap =
        if (src.config == Bitmap.Config.ARGB_8888) src
        else src.copy(Bitmap.Config.ARGB_8888, false)

    private fun toast(msg: String) = Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()

    override fun onDestroy() {
        super.onDestroy()
        cameraProvider?.unbindAll()
        inferExecutor.execute { detector.close() }
        inferExecutor.shutdown()
    }

    companion object {
        private const val TAG = "YoloMnn"
    }
}
