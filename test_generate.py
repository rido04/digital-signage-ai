from app.services.ai_generator import get_generator
from PIL import Image

print("🎨 Testing DirectML generation...")

# Get generator (will load model with DirectML)
generator = get_generator()

print("\n📊 Generator info:")
print(generator.get_model_info())

print("\n🚀 Starting generation (this will take ~3-5 minutes)...")

# Generate test person
image = generator.generate_person(
    template_name="chair_empty",
    pose="sitting",
    width=512,  # Small size for quick test
    height=768,
    num_inference_steps=20,  # Lower steps for speed
    guidance_scale=7.5,
    detected_gender="male",
    detected_age=30,
    template_base_prompt="full body photo of {gender} person in {pose} position, wearing {clothing}, hands resting naturally on lap, plain light gray background, no furniture visible, studio portrait, photorealistic, sharp focus, correct human anatomy, single person",
    template_negative_prompt="extra limbs, duplicate arms, duplicate legs, multiple hands, four arms, four legs, bad anatomy, deformed, multiple people, low quality, blurry, cartoon, anime, oversized head, chibi",
    generation_settings={"clothing_styles": ["casual"]}
)

if image:
    image.save("test_output_directml.jpg")
    print("\n✅ SUCCESS! Image saved to test_output_directml.jpg")
    print(f"📐 Size: {image.size}")
    print("\n🎉 DirectML generation works!")
else:
    print("\n❌ Generation failed!")