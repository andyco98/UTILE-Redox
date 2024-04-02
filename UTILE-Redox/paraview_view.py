from paraview.simple import *

# Load the volume data
volume_vti = XMLImageDataReader(FileName='C:/Users/a.colliard/Desktop/zeis_imgs/volume.vti')


# Create a color transfer function
colorTransferFunction = GetColorTransferFunction('Scalars_')
colorTransferFunction.RGBPoints = [0, 1.0, 1.0, 0.0,  # Yellow
                                   1, 1.0, 0.0, 0.0,  # Red
                                   2, 0.0, 0.0, 1.0,  # Blue
                                   3, 0.0, 1.0, 0.0,  # Green
                                   5, 0.0, 0.0, 0.0]  # Black

# Create an opacity transfer function
opacityTransferFunction = GetOpacityTransferFunction('Scalars_')
opacityTransferFunction.Points = [0, 0.001, 0.5, 0.0,
                                  1, 1.0, 0.5, 0.0,
                                  2, 0.2, 0.5, 0.0,
                                  3, 0.2, 0.5, 0.0,
                                  5, 0.0, 0.5, 0.0]

# Get active view
renderView = GetActiveViewOrCreate('RenderView')

# Update the volume representation
volume = Show(volume_vti, renderView)
volume.Representation = 'Volume'
volume.ColorArrayName = ['POINTS', 'Scalars_']  # Set the correct scalar property name
volume.LookupTable = colorTransferFunction
volume.ScalarOpacityFunction = opacityTransferFunction

# Update the view
renderView.Update()

# Start the interaction (uncomment if running as a standalone script)
#interact()
