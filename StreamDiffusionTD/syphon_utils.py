import numpy as np
import syphon
from syphon.server_directory import SyphonServerDirectory
import Metal
import objc
import warnings

# Suppress ObjCPointer warnings
warnings.filterwarnings("ignore", category=objc.ObjCPointerWarning)

class SyphonClientWrapper:
    def __init__(self, native_client, width=512, height=512, debug=False):
        self._client = native_client
        self.width = width
        self.height = height
        self.debug = debug
        
        # Create Metal device for texture handling
        import Metal
        self.device = Metal.MTLCreateSystemDefaultDevice()
        self.command_queue = self.device.newCommandQueue()
        
        if self.debug:
            print(f"\nInitialized with Metal device: {self.device}")
            print(f"Expected dimensions: {width}x{height}")
        
        # Get server description
        desc = self._client.serverDescription()
        if desc and self.debug:
            print("\nServer Description:")
            for key in desc.allKeys():
                print(f"- {key}: {desc[key]}")
        
    @property
    def has_new_frame(self):
        try:
            if self._client is None:
                return False
                
            # Get server description before checking frame
            desc = self._client.serverDescription()
            if desc and self.debug:
                print("\nCurrent Server State:")
                for key in desc.allKeys():
                    print(f"- {key}: {desc[key]}")
            
            result = self._client.hasNewFrame()
            if self.debug:
                print(f"Has new frame: {result}")
            
            # Try to get surface info - IMPORTANT: Keep this regardless of debug mode
            try:
                surface = self._client.newSurface()
                if surface and self.debug:
                    print(f"Surface available: {surface}")
            except Exception as e:
                if self.debug:
                    print(f"Surface check error: {e}")
            
            return bool(result)
        except Exception as e:
            if self.debug:
                print(f"Error checking for new frame: {e}")
            return False
        
    @property
    def new_frame_image(self):
        """Get new frame as a numpy array directly to avoid texture leaks"""
        try:
            if self._client is None:
                return None
                
            import objc
            pool = objc.autorelease_pool()
            with pool:
                if self.debug:
                    print("\nTrying to get frame via IOSurface...")
                
                # Get the IOSurface
                surface = self._client.newSurface()
                if surface is None:
                    if self.debug:
                        print("No IOSurface available")
                    return None
                    
                if self.debug:
                    print(f"Got IOSurface: {surface}")
                
                # Create a texture descriptor for the known dimensions
                descriptor = Metal.MTLTextureDescriptor.new()
                descriptor.setTextureType_(Metal.MTLTextureType2D)
                descriptor.setPixelFormat_(Metal.MTLPixelFormatBGRA8Unorm)
                descriptor.setWidth_(self.width)
                descriptor.setHeight_(self.height)
                descriptor.setStorageMode_(Metal.MTLStorageModeShared)  # Keep Shared for CPU access
                descriptor.setUsage_(Metal.MTLTextureUsageShaderRead)
                
                # Create a texture from the IOSurface
                try:
                    texture = self.device.newTextureWithDescriptor_iosurface_plane_(
                        descriptor,
                        surface,
                        0
                    )
                    
                    if texture is None:
                        if self.debug:
                            print("Failed to create texture from IOSurface")
                        return None
                        
                    if self.debug:
                        print(f"Created texture with dimensions: {texture.width()} x {texture.height()}")
                    
                    # Create a command buffer to synchronize the texture
                    command_buffer = self.command_queue.commandBuffer()
                    command_buffer.commit()
                    command_buffer.waitUntilCompleted()
                    
                    # Convert texture to numpy array
                    width = texture.width()
                    height = texture.height()
                    bytes_per_row = width * 4
                    buffer = bytearray(height * bytes_per_row)
                    
                    region = Metal.MTLRegionMake2D(0, 0, width, height)
                    texture.getBytes_bytesPerRow_fromRegion_mipmapLevel_(
                        buffer,
                        bytes_per_row,
                        region,
                        0
                    )
                    
                    # Release texture immediately
                    texture.setPurgeableState_(Metal.MTLPurgeableStateEmpty)
                    
                    # Return numpy array
                    frame = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 4)
                    return frame
                    
                except Exception as e:
                    if self.debug:
                        print(f"Error creating texture from IOSurface: {e}")
                        import traceback
                        print(traceback.format_exc())
                    return None
                
        except Exception as e:
            if self.debug:
                print(f"Error getting frame image: {e}")
                import traceback
                print(traceback.format_exc())
            return None

    def stop(self):
        try:
            if self._client is not None:
                self._client.stop()
                self._client = None
            
            # Release Metal resources
            if hasattr(self, 'command_queue'):
                self.command_queue.release()
                self.command_queue = None
        except Exception as e:
            if self.debug:
                print(f"Error stopping client: {e}")

class SyphonUtils:
    def __init__(self, sender_name="StreamDiffusion", input_name=None, control_name=None, width=512, height=512, debug=False):
        if debug:
            print(f"\n=== Initializing SyphonUtils ===")
            print(f"Sender name: {sender_name}")
            print(f"Input name: {input_name}")
            print(f"Control name: {control_name}")
            print(f"Dimensions: {width}x{height}")
        
        self.sender_name = sender_name
        self.input_name = input_name
        self.control_name = control_name
        self.width = width
        self.height = height
        self.debug = debug
        
        self.server = None
        self.input_client = None
        self.control_client = None
        self.directory = SyphonServerDirectory()

    def start(self):
        try:
            # List all available Syphon servers
            servers = self.directory.servers
            if self.debug:
                print(f"\n=== Available Syphon Servers ===")
                for server in servers:
                    print(f"App Name: {server.app_name}")
                    print(f"Name: {server.name}")
                    print(f"UUID: {server.uuid}")
                    print("---")

            # Initialize server
            self.server = syphon.SyphonMetalServer(self.sender_name)
            if self.debug:
                print(f"=== Syphon Server Initialized ===")
                print(f"Name: {self.sender_name}")
                print(f"Device: {self.server.device}")

            if self.input_name:
                self._connect_client('input')
            if self.control_name:
                self._connect_client('control')

        except Exception as e:
            if self.debug:
                print(f"Error initializing Syphon: {e}")

    def transmit_frame(self, frame):
        if self.server is None:
            return

        try:
            # Fix frame shape if it's (1, H, W, C)
            if len(frame.shape) == 4 and frame.shape[0] == 1:
                frame = frame[0]  # Remove batch dimension
                if self.debug:
                    print(f"Removed batch dimension. New shape: {frame.shape}")

            # Ensure frame is contiguous and RGBA
            frame = np.ascontiguousarray(frame)
            if frame.shape[2] == 3:
                alpha = np.full((frame.shape[0], frame.shape[1], 1), 255, dtype=np.uint8)
                frame = np.concatenate([frame, alpha], axis=2)
                frame = np.ascontiguousarray(frame)  # Ensure the concatenated array is contiguous

            # Create texture
            descriptor = Metal.MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
                Metal.MTLPixelFormatRGBA8Unorm,
                frame.shape[1],  # width
                frame.shape[0],  # height
                False
            )
            descriptor.setStorageMode_(Metal.MTLStorageModeShared)
            descriptor.setUsage_(Metal.MTLTextureUsageShaderRead)
            
            texture = self.server.device.newTextureWithDescriptor_(descriptor)
            
            # Copy frame data
            region = Metal.MTLRegionMake2D(0, 0, frame.shape[1], frame.shape[0])
            frame_bytes = frame.tobytes()
            bytes_per_row = frame.shape[1] * 4  # RGBA = 4 bytes per pixel
            
            texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
                region,
                0,
                frame_bytes,
                bytes_per_row
            )
            
            # Publish
            self.server.publish_frame_texture(texture, is_flipped=True)
            
            # Mark texture as purgeable to help with memory management
            texture.setPurgeableState_(Metal.MTLPurgeableStateEmpty)
            
            if self.debug:
                print(f"Transmitted frame via Syphon")

        except Exception as e:
            if self.debug:
                print(f"Error transmitting Syphon frame: {e}")
                import traceback
                print(traceback.format_exc())
            
    def _connect_client(self, client_type='input'):
        name = self.input_name if client_type == 'input' else self.control_name
        try:
            if self.debug:
                print(f"\n=== Connecting {client_type} Client ===")
                print(f"Looking for server: {name}")
            
            import objc
            SyphonServerDirectory = objc.lookUpClass('SyphonServerDirectory')
            directory = SyphonServerDirectory.sharedDirectory()
            
            # Get servers directly from Objective-C
            servers = directory.servers()
            
            for server_desc in servers:
                app_name = str(server_desc.objectForKey_('SyphonServerDescriptionAppNameKey') or '')
                server_name = str(server_desc.objectForKey_('SyphonServerDescriptionNameKey') or '')
                
                if self.debug:
                    print(f"Checking server: {app_name} - {server_name}")
                
                if app_name == name or server_name == name:
                    if self.debug:
                        print(f"\nFound Matching Server: {server_name}")
                    
                    try:
                        # Create the client using Objective-C bridge directly
                        SyphonClient = objc.lookUpClass('SyphonClient')
                        native_client = SyphonClient.alloc().initWithServerDescription_options_newFrameHandler_(
                            server_desc,
                            None,
                            None
                        )
                        
                        if native_client:
                            wrapped_client = SyphonClientWrapper(
                                native_client,
                                width=self.width,
                                height=self.height,
                                debug=self.debug
                            )
                            
                            if client_type == 'input':
                                if self.input_client is not None:
                                    self.input_client.stop()
                                self.input_client = wrapped_client
                            else:
                                if self.control_client is not None:
                                    self.control_client.stop()
                                self.control_client = wrapped_client
                                
                            if self.debug:
                                print(f"Connected to {client_type} Syphon server: {name}")
                            return True
                        else:
                            if self.debug:
                                print(f"Failed to create client for: {name}")
                    
                    except Exception as e:
                        if self.debug:
                            print(f"\n=== Client Creation Error ===")
                            print(f"Error: {e}")
                            import traceback
                            print(traceback.format_exc())
            
            if self.debug:
                print(f"Warning: No matching Syphon server found for name: {name}")
            return False
                
        except Exception as e:
            if self.debug:
                print(f"Error connecting to {client_type} Syphon server: {e}")
                import traceback
                print(traceback.format_exc())
            return False

    def capture_input_frame(self):
        if self.input_client is None:
            return None

        try:
            if not self.input_client.has_new_frame:
                if self.debug:
                    print("No new input frame available")
                return None

            if self.debug:
                print("\n=== Capturing Input Frame ===")
            
            # Get frame directly as numpy array to avoid texture leaks
            frame = self.input_client.new_frame_image
            if frame is None:
                return None
            
            # Convert RGBA to RGB
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]
                if self.debug:
                    print(f"Converted to RGB. New shape: {frame.shape}")
            
            return frame
                
        except Exception as e:
            if self.debug:
                print(f"Error capturing input frame: {e}")
                import traceback
                print(traceback.format_exc())
            return None

    def capture_control_frame(self):
        if self.control_client is None:
            return None

        try:
            if not self.control_client.has_new_frame:
                if self.debug:
                    print("No new control frame available")
                return None

            if self.debug:
                print("\n=== Capturing Control Frame ===")
            
            # Get frame directly as numpy array to avoid texture leaks
            frame = self.control_client.new_frame_image
            if frame is None:
                return None
            
            # Convert RGBA to RGB
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]
                if self.debug:
                    print(f"Converted to RGB. New shape: {frame.shape}")
            
            return frame
            
        except Exception as e:
            if self.debug:
                print(f"Error capturing control frame: {e}")
                import traceback
                print(traceback.format_exc())
            return None

    def stop(self):
        if self.debug:
            print("\n=== Stopping Syphon ===")
        if self.server:
            self.server.stop()
            self.server = None
            if self.debug:
                print("Server stopped")
        if self.input_client:
            self.input_client.stop()
            self.input_client = None
            if self.debug:
                print("Input client stopped")
        if self.control_client:
            self.control_client.stop()
            self.control_client = None
            if self.debug:
                print("Control client stopped")