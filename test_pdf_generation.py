#!/usr/bin/env python3

import os
from enhanced_drawing_analyzer import EnhancedDrawingAnalyzer
from PIL import Image, ImageDraw

def test_pdf_generation():
    """Test PDF generation specifically"""
    print("🧪 Testing PDF generation...")
    
    # Create a simple test image (like your house drawing)
    test_image = Image.new('RGB', (400, 400), color='white')
    draw = ImageDraw.Draw(test_image)
    
    # Draw a simple house (similar to your test)
    draw.rectangle([100, 200, 300, 350], fill='brown', outline='black')  # House
    draw.polygon([(80, 200), (200, 100), (320, 200)], fill='red')  # Roof
    draw.rectangle([150, 250, 200, 350], fill='blue')  # Door
    draw.rectangle([220, 250, 270, 300], fill='yellow')  # Window
    draw.ellipse([50, 50, 100, 100], fill='yellow')  # Sun
    
    test_image.save('test_pdf_drawing.png')
    print("🏠 Created test drawing")
    
    # Test the analyzer with PDF generation
    try:
        analyzer = EnhancedDrawingAnalyzer()
        results = analyzer.analyze_drawing_with_pdf_report(
            'test_pdf_drawing.png', 
            6, 
            "House Drawing",
            generate_pdf=True
        )
        
        if results and 'error' not in results:
            print("✅ Analysis completed successfully!")
            
            if 'pdf_report_filename' in results and results['pdf_report_filename']:
                pdf_file = results['pdf_report_filename']
                if os.path.exists(pdf_file):
                    file_size = os.path.getsize(pdf_file)
                    print(f"✅ PDF generated successfully: {pdf_file}")
                    print(f"📄 PDF file size: {file_size:,} bytes")
                    print(f"🎯 You can now download this PDF in your Streamlit app!")
                else:
                    print(f"❌ PDF file not found: {pdf_file}")
            else:
                print("❌ No PDF filename in results")
                
        else:
            print(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install reportlab pillow")

if __name__ == "__main__":
    test_pdf_generation()

