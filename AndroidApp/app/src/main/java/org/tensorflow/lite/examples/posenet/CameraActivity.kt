package org.tensorflow.lite.examples.posenet

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

class CameraActivity : AppCompatActivity() {

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.tfe_pn_activity_camera)
    savedInstanceState ?: supportFragmentManager.beginTransaction()
      .replace(R.id.container, PosenetActivity())
      .commit()
  }
}
