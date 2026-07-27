package com.yolomnn.demo

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream

/** CameraX ImageProxy -> upright RGB Bitmap conversion helpers. */
object ImageUtils {

    /** Convert a YUV_420_888 ImageProxy to a Bitmap, applying rotation. */
    fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        val nv21 = yuv420ToNv21(image)
        val yuv = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuv.compressToJpeg(Rect(0, 0, image.width, image.height), 85, out)
        val bytes = out.toByteArray()
        val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        val rotation = image.imageInfo.rotationDegrees
        if (rotation == 0) return bmp
        val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
        val rotated = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, matrix, true)
        if (rotated !== bmp) bmp.recycle()
        return rotated
    }

    private fun yuv420ToNv21(image: ImageProxy): ByteArray {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()
        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        // NV21 expects V then U interleaved; copying V buffer then U buffer works
        // for the common semi-planar case used by CameraX JPEG re-encode path.
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)
        return nv21
    }
}
