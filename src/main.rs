use std::collections::HashMap;
use std::fmt;

mod vk_format;

pub use vk_format::VkFormat;

fn main() -> anyhow::Result<()> {
    #[rustfmt::skip]
    let file = &[
        // Header
        0xAB, 0x4B, 0x54, 0x58, // first four bytes of Byte[12] identifier
        0x20, 0x32, 0x30, 0xBB, // next four bytes of Byte[12] identifier
        0x0D, 0x0A, 0x1A, 0x0A, // final four bytes of Byte[12] identifier
        0x00, 0x00, 0x00, 0x00, // UInt32 vkFormat = VK_FORMAT_UNDEFINED (0)
        0x01, 0x00, 0x00, 0x00, // UInt32 typeSize = 1
        0x08, 0x00, 0x00, 0x00, // UInt32 pixelWidth = 8
        0x08, 0x00, 0x00, 0x00, // UInt32 pixelHeight = 8
        0x00, 0x00, 0x00, 0x00, // UInt32 pixelDepth = 0
        0x00, 0x00, 0x00, 0x00, // UInt32 layerCount = 0
        0x01, 0x00, 0x00, 0x00, // UInt32 faceCount = 0
        0x01, 0x00, 0x00, 0x00, // UInt32 levelCount = 0
        0x01, 0x00, 0x00, 0x00, // UInt32 supercompressionScheme = 1 (BASISLZ)
        // Index
        0x68, 0x00, 0x00, 0x00, // Uint32 dfdByteOffset = 0x00000068
        0x5C, 0x00, 0x00, 0x00, // UInt32 dfdByteSize = 0x0000005C
        0xC4, 0x00, 0x00, 0x00, // UInt32 kvdByteOffset = 0x000000C4
        0x58, 0x00, 0x00, 0x00, // UInt32 kvdByteLength = 0x00000058
        0x20, 0x01, 0x00, 0x00, // UInt64 sgdByteOffset = 0x0000000000000120
        0x00, 0x00, 0x00, 0x00,
        0x90, 0x00, 0x00, 0x00, // UInt64 sgdByteLength = 0x0000000000000090
        0x00, 0x00, 0x00, 0x00,
        // Level Index
        0xB0, 0x01, 0x00, 0x00, // UInt64 level[0].byteOffset = 0x00000000000001B0
        0x00, 0x00, 0x00, 0x00,
        0x03, 0x00, 0x00, 0x00, // UInt64 level[0].byteLength = 0x0000000000000003
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, // UInt64 level[0].uncompressedByteLength = 0
        0x00, 0x00, 0x00, 0x00,
        // DFD
        0x3C, 0x00, 0x00, 0x00, // UInt32 dfdTotalSize = 0x3C (60)
        0x00, 0x00, 0x00, 0x00, // vendorId = 0 (17 bits), descriptorType = 0
        0x02, 0x00, 0x38, 0x00, // versionNumber = 2, descriptorBlockSize = 0x38 (56)
        0xA3, 0x01, 0x02, 0x00, // colorModel = ETC1S (163), primaries = BT709 (1)
                                // transferFunction = SRGB (2), flags = 0
        0x03, 0x03, 0x00, 0x00, // texelBlockDimension[[0-3] = 3, 3, 0, 0
        0x00, 0x00, 0x00, 0x00, // bytesPlane[0-3] = 0
        0x00, 0x00, 0x00, 0x00, // bytesPlane[4-7] = 0
        // DFD sample information, sample 0
        0x00, 0x00, 0x3F, 0x00, // bitOffset = 0 bitLength = 0x3F (63),
                                // channelType = RGB (0), qualifiers = 0
        0x00, 0x00, 0x00, 0x00, // samplePosition[0-3] = 0
        0x00, 0x00, 0x00, 0x00, // sampleLower = 0
        0xFF, 0xFF, 0xFF, 0xFF, // sampleUpper = 0xFFFFFFFF (UINT_MAX)
        // Sample 1
        0x40, 0x00, 0x3F, 0x0F, // bitOffset = 0x40 (64) bitLength = 0x3F (63),
                                // channelType = AAA (0x0F), qualifiers = 0
        0x00, 0x00, 0x00, 0x00, // samplePosition[0-3] = 0
        0x00, 0x00, 0x00, 0x00, // sampleLower = 0
        0xFF, 0xFF, 0xFF, 0xFF, // sampleUpper = 0xFFFFFFFF (UINT_MAX)
        // Key/Value Data
        0x12, 0x00, 0x00, 0x00, // keyAndValueByteLength = 18 (0x12)
        0x4B, 0x54, 0x58, 0x6F, // KTXo
        0x72, 0x69, 0x65, 0x6E, // rien
        0x74, 0x61, 0x74, 0x69, // tati
        0x6F, 0x6E, 0x00, 0x72, // on NUL r
        0x64, 0x00, 0x00, 0x00, // d  <3 bytes of valuePadding>
        0x3B, 0x00, 0x00, 0x00, // keyAndValueByteLength = 59 (0x3B)
        0x4B, 0x54, 0x58, 0x77, // KTXw
        0x72, 0x69, 0x74, 0x65, // rite
        0x72, 0x00, 0x74, 0x6F, // r NUL to
        0x6B, 0x74, 0x78, 0x20, // ktx SPACE
        0x76, 0x34, 0x2E, 0x30, // v4.0
        0x2E, 0x5F, 0x5F, 0x64, // .__d
        0x65, 0x66, 0x61, 0x75, // efau
        0x6C, 0x74, 0x5F, 0x5F, // lt__
        0x20, 0x2F, 0x20, 0x6C, // SPACE / SPACE l
        0x69, 0x62, 0x6B, 0x74, // ibkt
        0x78, 0x20, 0x76, 0x34, // x v4
        0x2E, 0x30, 0x2E, 0x5F, // .0._
        0x5F, 0x64, 0x65, 0x66, // _def
        0x61, 0x75, 0x6C, 0x74, // ault
        0x5F, 0x5F, 0x00, 0x00, // __ <2 bytes of valuePadding>
        0x00, 0x00, 0x00, 0x00, // 4 bytes of padding.
        // Supercompression Global Data
        0x02, 0x00, 0x02, 0x00, // UInt16 endpointCount = 2, UInt16 selectorCount = 2
        0x2D, 0x00, 0x00, 0x00, // UInt32 endpointsByteLength = 0x2D
        0x09, 0x00, 0x00, 0x00, // UInt32 selectorsByteLength = 0x09
        0x2E, 0x00, 0x00, 0x00, // Uint32 tablesByteLength = 0x2E
        0x00, 0x00, 0x00, 0x00, // Uint32 extendedByteLength = 0
        // imageDesc[0]
        0x00, 0x00, 0x00, 0x00, // UInt32 flags = 0
        0x00, 0x00, 0x00, 0x00, // UInt32 rgbSliceByteOffset = 0
        0x02, 0x00, 0x00, 0x00, // UInt32 rgbSliceByteLength = 2
        0x02, 0x00, 0x00, 0x00, // UInt32 alphaSliceByteOffset = 0x02
        0x01, 0x00, 0x00, 0x00, // UInt32 alphaSliceByteLength = 1
        // endpointsData
        0x01, 0xC0, 0x04, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x02, 0x04, 0x98,
        0x1B, 0x20, 0x00, 0x00,
        0x00, 0x08, 0xC3, 0x36,
        0x91, 0x3E, 0x91, 0x00,
        0x60, 0x02, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x81, 0x00, 0x4C, 0x01,
        0x10, 0x00, 0x00, 0x00,
        0x00, 0x20, 0x59, 0xC0,
        0x3D,
        // selectorsData
            0x54, 0x55, 0x55,
        0x55, 0xAD, 0xAA, 0xAA,
        0xAA, 0x02,
        // tablesData
                    0x14, 0xC0,
        0x44, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x12,
        0x41, 0x00, 0x98, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x40, 0x18, 0x02,
        0xA2, 0x04, 0x0C, 0x00,
        0x00, 0x00, 0x83, 0x76,
        0x7B, 0x49, 0x04, 0xA2,
        0x20, 0x00, 0x4C, 0x00,
        0x08, 0x00, 0x00, 0x00,
        0x00, 0x20, 0x02, 0x01,
        // Level 0 image data
        0x4E, 0x0E, 0x04
    ][..];

    for file in std::env::args().skip(1) {
        dbg!(&file);
        let file = std::fs::File::open(file).unwrap();

        let ktx2 = Reader::new(file)?;

        if ktx2.header.vk_format == VkFormat::UNDEFINED {
            if ktx2.vk_format() == VkFormat::UNDEFINED {
                dbg!(
                    &ktx2.header.supercompression_scheme,
                    &ktx2.descriptor_block,
                    &ktx2.samples,
                    &ktx2.key_values.swizzle
                );
                dbg!(ktx2.vk_format());
            }
        }
    }

    Ok(())
}

#[derive(Debug)]
struct Header {
    vk_format: VkFormat,
    type_size: u32,
    pixel_width: u32,
    pixel_height: u32,
    pixel_depth: u32,
    layer_count: u32,
    face_count: u32,
    level_count: u32,
    supercompression_scheme: SupercompressionScheme,
}

#[derive(Debug)]
struct Index {
    dfd_byte_offset: u32,
    dfd_byte_length: u32,
    kvd_byte_offset: u32,
    kvd_byte_length: u32,
    sgd_byte_offset: u64,
    sgd_byte_length: u64,
}

#[derive(Debug)]
struct Level {
    byte_offset: u64,
    byte_length: u64,
    uncompressed_byte_length: u64,
}

struct Reader<T> {
    read_bytes: usize,
    inner: T,
}

impl<T: std::io::Read> Reader<T> {
    fn new(inner_reader: T) -> anyhow::Result<Ktx2> {
        let mut this = Self {
            read_bytes: 0,
            inner: inner_reader,
        };

        let magic = [
            b'\xAB', b'K', b'T', b'X', b' ', b'2', b'0', b'\xBB', b'\r', b'\n', b'\x1A', b'\n',
        ];

        assert_eq!(this.read_array::<12>()?, magic);

        let header = Header {
            vk_format: VkFormat(this.read_u32()?),
            type_size: this.read_u32()?,
            pixel_width: this.read_u32()?,
            pixel_height: this.read_u32()?,
            pixel_depth: this.read_u32()?,
            layer_count: this.read_u32()?,
            face_count: this.read_u32()?,
            level_count: this.read_u32()?,
            supercompression_scheme: SupercompressionScheme::parse(this.read_u32()?),
        };

        let index = Index {
            dfd_byte_offset: this.read_u32()?,
            dfd_byte_length: this.read_u32()?,
            kvd_byte_offset: this.read_u32()?,
            kvd_byte_length: this.read_u32()?,
            sgd_byte_offset: this.read_u64()?,
            sgd_byte_length: this.read_u64()?,
        };

        let levels = (0..header.level_count.max(1))
            .map(|_| {
                Ok(Level {
                    byte_offset: this.read_u64()?,
                    byte_length: this.read_u64()?,
                    uncompressed_byte_length: this.read_u64()?,
                })
            })
            .collect::<std::io::Result<Vec<_>>>()?;

        let dfd_start = this.read_bytes;

        let dfd_total_size = this.read_u32()?;

        debug_assert_eq!(dfd_total_size, index.dfd_byte_length);

        let descriptor_block_header = DescriptorBlockHeader {
            vendor_id_and_descriptor_type: this.read_u32()?,
            version_number: this.read_u16()?,
            descriptor_block_size: this.read_u16()?,
        };

        let descriptor_block = DescriptorBlock {
            color_model: ColorModel(this.read_u8()?),
            color_primaries: ColorPrimaries::parse(this.read_u8()?).unwrap(),
            transfer_function: TransferFunction::parse(this.read_u8()?).unwrap(),
            flags: DescriptorBlockFlags {
                bits: this.read_u8()?,
            },
            texel_block_dimensions: this.read_array()?,
            bytes_plane: this.read_array()?,
        };

        let num_samples = (dfd_total_size as usize - (this.read_bytes - dfd_start))
            / std::mem::size_of::<RawSampleInformation>();

        let samples = (0..num_samples)
            .map(|_| {
                let sample = RawSampleInformation {
                    bit_offset: this.read_u16()?,
                    bit_length: this.read_u8()?,
                    channel_type_and_qualifiers: this.read_u8()?,
                    sample_position: this.read_array()?,
                    sample_lower: this.read_u32()?,
                    sample_upper: this.read_u32()?,
                };

                let qualifiers = SampleQualifiers {
                    bits: sample.channel_type_and_qualifiers >> 4,
                };

                let channel_type_bits = sample.channel_type_and_qualifiers & 0b00001111;

                Ok(SampleInformation {
                    channel_type: match descriptor_block.color_model {
                        ColorModel::ETC1S => {
                            ChannelType::Etc1S(Etc1SChannelType::parse(channel_type_bits).unwrap())
                        }
                        ColorModel::UASTC => {
                            ChannelType::Uastc(UastcChannelType::parse(channel_type_bits).unwrap())
                        }
                        _ => ChannelType::Regular(RegularChannelType(channel_type_bits)),
                    },
                    sample_position: sample.sample_position,
                    sample_lower: if qualifiers.contains(SampleQualifiers::FLOAT) {
                        SamplerBound::Float(f32::from_le_bytes(sample.sample_lower.to_le_bytes()))
                    } else {
                        SamplerBound::Uint(sample.sample_lower)
                    },
                    sample_upper: if qualifiers.contains(SampleQualifiers::FLOAT) {
                        SamplerBound::Float(f32::from_le_bytes(sample.sample_upper.to_le_bytes()))
                    } else {
                        SamplerBound::Uint(sample.sample_upper)
                    },
                    qualifiers,
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let kv_start_bytes = this.read_bytes;

        let mut key_values = HashMap::new();

        while this.read_bytes - kv_start_bytes < index.kvd_byte_length as usize {
            let (key, value) = this.read_key_value_pair()?;
            key_values.insert(key, value);
        }

        if index.sgd_byte_length > 0 {
            this.align(8)?;
        }

        if let SupercompressionScheme::BasisLZ = header.supercompression_scheme {
            let global_data_header = BasisLZGlobalDataHeader {
                endpoint_count: this.read_u16()?,
                selector_count: this.read_u16()?,
                endpoints_byte_length: this.read_u32()?,
                selectors_byte_length: this.read_u32()?,
                tables_byte_length: this.read_u32()?,
                extended_byte_length: this.read_u32()?,
            };

            let image_desc = BasisLZGlobalDataImageDesc {
                flags: ImageFlags {
                    bits: this.read_u32()?,
                },
                rgb_slice_byte_offset: this.read_u32()?,
                rgb_slice_byte_length: this.read_u32()?,
                alpha_slice_byte_offset: this.read_u32()?,
                alpha_slice_byte_length: this.read_u32()?,
            };

            let endpoints_data =
                this.read_vec(global_data_header.endpoints_byte_length as usize)?;
            let selectors_data =
                this.read_vec(global_data_header.selectors_byte_length as usize)?;
            let tables_data = this.read_vec(global_data_header.tables_byte_length as usize)?;
        }

        let start_of_levels = this.read_bytes;

        let mut level_data_length = 0;

        for level in &levels {
            level_data_length = level_data_length
                .max((level.byte_length + level.byte_offset) as usize - this.read_bytes);
        }

        let compressed_image_data = this.read_vec(level_data_length)?;
        let mut compressed_images = Vec::with_capacity(levels.len());

        for level in &levels {
            let offset = level.byte_offset as usize - start_of_levels;

            compressed_images.push(Vec::from(
                &compressed_image_data[offset..offset + level.byte_length as usize],
            ));
        }

        Ok(Ktx2 {
            header,
            compressed_images,
            key_values: KeyValues::fetch(&key_values)?,
            custom_key_values: key_values,
            descriptor_block,
            samples,
        })
    }

    fn read_vec(&mut self, len: usize) -> std::io::Result<Vec<u8>> {
        let mut vec = vec![0; len];
        self.inner.read_exact(&mut vec)?;
        self.read_bytes += len;
        Ok(vec)
    }

    fn read_array<const LEN: usize>(&mut self) -> std::io::Result<[u8; LEN]> {
        let mut array = [0; LEN];
        self.inner.read_exact(&mut array)?;
        self.read_bytes += LEN;
        Ok(array)
    }

    fn read_u8(&mut self) -> std::io::Result<u8> {
        Ok(u8::from_le_bytes(self.read_array()?))
    }

    fn read_u16(&mut self) -> std::io::Result<u16> {
        Ok(u16::from_le_bytes(self.read_array()?))
    }

    fn read_u32(&mut self) -> std::io::Result<u32> {
        Ok(u32::from_le_bytes(self.read_array()?))
    }

    fn read_u64(&mut self) -> std::io::Result<u64> {
        Ok(u64::from_le_bytes(self.read_array()?))
    }

    fn align(&mut self, bytes: usize) -> std::io::Result<()> {
        let remainder = self.read_bytes % bytes;

        if remainder > 0 {
            let to_read = bytes - remainder;
            self.read_vec(to_read)?;
        }

        Ok(())
    }

    fn read_key_value_pair(&mut self) -> std::io::Result<(String, String)> {
        let length = self.read_u32()?;
        let key_and_value = self.read_vec(length as usize)?;

        let key_end_index = key_and_value.iter().position(|&c| c == b'\0').unwrap();
        let key =
            unsafe { std::str::from_utf8_unchecked(&key_and_value[..key_end_index as usize]) };
        let value = unsafe {
            std::str::from_utf8_unchecked(
                &key_and_value[key_end_index as usize + 1..key_and_value.len() - 1],
            )
        };

        self.align(4)?;

        Ok((key.into(), value.into()))
    }
}

#[derive(Debug)]
struct KeyValues {
    writer: Option<String>,
    writer_params: Option<String>,
    orientation: Option<Orientation>,
    swizzle: Option<[SwizzleComponent; 4]>,
}

impl KeyValues {
    fn fetch(map: &HashMap<String, String>) -> anyhow::Result<Self> {
        Ok(Self {
            writer: map.get("KTXwriter").cloned(),
            writer_params: map.get("KTXwriterScParams").cloned(),
            orientation: match map.get("KTXorientation") {
                Some(string) => Some(Orientation::parse(string)?),
                None => None,
            },
            swizzle: match map.get("KTXswizzle") {
                Some(string) => {
                    let mut chars = string.chars();

                    let mut component = || match chars.next() {
                        Some(character) => SwizzleComponent::parse(character).ok_or_else(|| {
                            anyhow::anyhow!("Invalid swizzle component: {}", string)
                        }),
                        None => Err(anyhow::anyhow!("Missing swizzle component")),
                    };

                    Some([component()?, component()?, component()?, component()?])
                }
                None => None,
            },
        })
    }
}

#[derive(Debug)]
enum Orientation {
    D1(HorizontalOrientation),
    D2(HorizontalOrientation, VerticalOrientation),
    D3(HorizontalOrientation, VerticalOrientation, DepthOrientation),
}

impl Orientation {
    fn parse(string: &str) -> anyhow::Result<Self> {
        let mut chars = string.chars();

        let horizontal = match chars.next() {
            Some('r') => HorizontalOrientation::Right,
            Some('l') => HorizontalOrientation::Left,
            Some(other) => return Err(anyhow::anyhow!("Unknown orientation char {}", other)),
            None => return Err(anyhow::anyhow!("Missing orientation char")),
        };

        let vertical = match chars.next() {
            Some('d') => VerticalOrientation::Down,
            Some('u') => VerticalOrientation::Up,
            Some(other) => return Err(anyhow::anyhow!("Unknown orientation char {}", other)),
            None => return Ok(Self::D1(horizontal)),
        };

        let depth = match chars.next() {
            Some('o') => DepthOrientation::Out,
            Some('i') => DepthOrientation::In,
            Some(other) => return Err(anyhow::anyhow!("Unknown orientation char {}", other)),
            None => return Ok(Self::D2(horizontal, vertical)),
        };

        if chars.next().is_some() {
            return Err(anyhow::anyhow!(
                "orientation string {} has too many chars",
                string
            ));
        }

        Ok(Self::D3(horizontal, vertical, depth))
    }
}

#[derive(Debug)]
enum SwizzleComponent {
    Red,
    Green,
    Blue,
    Alpha,
    Zero,
    One,
}

impl SwizzleComponent {
    fn parse(character: char) -> Option<Self> {
        match character {
            'r' => Some(Self::Red),
            'g' => Some(Self::Green),
            'b' => Some(Self::Blue),
            'a' => Some(Self::Alpha),
            '0' => Some(Self::Zero),
            '1' => Some(Self::One),
            _ => None,
        }
    }
}

#[derive(Debug)]
enum HorizontalOrientation {
    Right,
    Left,
}

#[derive(Debug)]
enum VerticalOrientation {
    Down,
    Up,
}

#[derive(Debug)]
enum DepthOrientation {
    Out,
    In,
}

struct Ktx2 {
    header: Header,
    descriptor_block: DescriptorBlock,
    compressed_images: Vec<Vec<u8>>,
    key_values: KeyValues,
    custom_key_values: HashMap<String, String>,
    samples: Vec<SampleInformation>,
}

impl Ktx2 {
    fn images_decompressed(&self) -> impl Iterator<Item = anyhow::Result<Vec<u8>>> + '_ {
        self.compressed_images
            .iter()
            .map(|image| match &self.header.supercompression_scheme {
                SupercompressionScheme::None => Ok(image.clone()),
                SupercompressionScheme::Zstandard => Ok(zstd::stream::decode_all(&image[..])?),
                scheme => panic!("Not supported: {:?}", scheme),
            })
    }

    fn vk_format(&self) -> VkFormat {
        if self.header.vk_format != VkFormat::UNDEFINED {
            self.header.vk_format
        } else {
            let mut num_channels = 0;

            for sample in &self.samples {
                if sample.qualifiers != SampleQualifiers::NONE {
                    dbg!(sample.qualifiers);
                }

                match sample.channel_type {
                    ChannelType::Etc1S(Etc1SChannelType::Rrr)
                    | ChannelType::Uastc(UastcChannelType::Rrr) => {
                        num_channels = num_channels.max(1)
                    }
                    ChannelType::Etc1S(Etc1SChannelType::Ggg)
                    | ChannelType::Uastc(UastcChannelType::Rrrg)
                    | ChannelType::Uastc(UastcChannelType::Rg) => {
                        num_channels = num_channels.max(2)
                    }
                    ChannelType::Etc1S(Etc1SChannelType::Rgb)
                    | ChannelType::Uastc(UastcChannelType::Rgb) => {
                        num_channels = num_channels.max(3)
                    }
                    ChannelType::Uastc(UastcChannelType::Rgba)
                    | ChannelType::Etc1S(Etc1SChannelType::Aaa) => {
                        num_channels = num_channels.max(4)
                    }
                    ChannelType::Regular(_) => {
                        // Not reachable?
                    }
                }
            }

            match num_channels {
                1 => VkFormat::R8_UNORM,
                2 => VkFormat::R8G8_UNORM,
                3 => VkFormat::R8G8B8_UNORM,
                4 => VkFormat::R8G8B8A8_UNORM,
                _ => unreachable!(),
            }
        }
    }
}

#[derive(Debug)]
struct DescriptorBlockHeader {
    // vendor id is 17 bits
    // https://www.khronos.org/registry/DataFormat/specs/1.3/dataformat.1.3.html#_anchor_id_descriptorblock_xreflabel_descriptorblock_descriptor_block
    vendor_id_and_descriptor_type: u32,
    version_number: u16,
    descriptor_block_size: u16,
}

#[derive(Debug)]
struct DescriptorBlock {
    color_model: ColorModel,
    color_primaries: ColorPrimaries,
    transfer_function: TransferFunction,
    flags: DescriptorBlockFlags,
    texel_block_dimensions: [u8; 4],
    bytes_plane: [u8; 8],
}

#[derive(Debug)]
enum SupercompressionScheme {
    None = 0,
    BasisLZ = 1,
    Zstandard = 2,
    ZLIB = 3,
    Reserved = 4,
}

impl SupercompressionScheme {
    fn parse(value: u32) -> Self {
        match value {
            0 => Self::None,
            1 => Self::BasisLZ,
            2 => Self::Zstandard,
            3 => Self::ZLIB,
            _ => Self::Reserved,
        }
    }
}

bitflags::bitflags! {
    struct DescriptorBlockFlags: u8 {
        const ALPHA_STRAIGHT = 0b0;
        const ALPHA_PREMULTIPLIED = 0b1;
    }
}

struct RawSampleInformation {
    bit_offset: u16,
    bit_length: u8,
    channel_type_and_qualifiers: u8,
    sample_position: [u8; 4],
    sample_lower: u32,
    sample_upper: u32,
}

#[derive(Debug)]
struct SampleInformation {
    channel_type: ChannelType,
    qualifiers: SampleQualifiers,
    sample_position: [u8; 4],
    sample_upper: SamplerBound,
    sample_lower: SamplerBound,
}

#[derive(Debug)]
enum SamplerBound {
    Uint(u32),
    Float(f32),
}

#[derive(Debug)]
struct BasisLZGlobalDataHeader {
    endpoint_count: u16,
    selector_count: u16,
    endpoints_byte_length: u32,
    selectors_byte_length: u32,
    tables_byte_length: u32,
    extended_byte_length: u32,
}

#[derive(Debug)]
struct BasisLZGlobalDataImageDesc {
    flags: ImageFlags,
    rgb_slice_byte_offset: u32,
    rgb_slice_byte_length: u32,
    alpha_slice_byte_offset: u32,
    alpha_slice_byte_length: u32,
}

bitflags::bitflags! {
    struct ImageFlags: u32 {
        const NONE = 0b0;
        const IS_P_FRAME = 0x02;
    }
}

#[derive(PartialEq, Eq)]
pub struct ColorModel(u8);

impl ColorModel {
    pub const RGBSDA: Self = Self(1);
    pub const BC2: Self = Self(129);
    pub const BC3: Self = Self(130);
    pub const ETC2: Self = Self(161);
    pub const ASTC: Self = Self(162);
    pub const ETC1S: Self = Self(163);
    pub const UASTC: Self = Self(166);
}

impl fmt::Debug for ColorModel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let name = match *self {
            Self::RGBSDA => Some("RGBSDA"),
            Self::BC2 => Some("BC2"),
            Self::BC3 => Some("BC3"),
            Self::ETC2 => Some("ETC2"),
            Self::ASTC => Some("ASTC"),
            Self::ETC1S => Some("ETC1S"),
            Self::UASTC => Some("UASTC"),
            _ => None,
        };

        if let Some(x) = name {
            f.write_str(x)
        } else {
            self.0.fmt(f)
        }
    }
}

#[derive(Debug)]
enum TransferFunction {
    Unspecified = 0,
    Linear = 1,
    SRGB = 2,
}

impl TransferFunction {
    fn parse(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Unspecified),
            1 => Some(Self::Linear),
            2 => Some(Self::SRGB),
            _ => None,
        }
    }
}

bitflags::bitflags! {
    // https://www.khronos.org/registry/DataFormat/specs/1.3/dataformat.1.3.html#SampleOverview
    struct SampleQualifiers: u8 {
        const NONE = 0;
        const FLOAT = 8;
        const SIGNED = 4;
        const EXPONENT = 2;
        const LINEAR = 1;
    }
}

#[derive(Debug)]
enum ChannelType {
    Regular(RegularChannelType),
    Etc1S(Etc1SChannelType),
    Uastc(UastcChannelType),
}

#[derive(PartialEq, Eq)]
struct RegularChannelType(u8);

// https://www.khronos.org/registry/DataFormat/specs/1.3/dataformat.1.3.html#RGBSDAChannels
// extended by:
// https://github.khronos.org/KTX-Specification/#etc1s
impl RegularChannelType {
    pub const RED: Self = Self(0);
    pub const GREEN: Self = Self(1);
    pub const BLUE: Self = Self(2);
    pub const RRR: Self = Self(3);
    pub const GGG: Self = Self(4);
    pub const STENCIL: Self = Self(13);
    pub const DEPTH: Self = Self(14);
    pub const ALPHA: Self = Self(15);
}

impl fmt::Debug for RegularChannelType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let name = match *self {
            Self::RED => Some("RED"),
            Self::GREEN => Some("GREEN"),
            Self::BLUE => Some("BLUE"),
            Self::RRR => Some("RRR"),
            Self::GGG => Some("GGG"),
            Self::STENCIL => Some("STENCIL"),
            Self::DEPTH => Some("DEPTH"),
            Self::ALPHA => Some("ALPHA"),
            _ => None,
        };

        if let Some(x) = name {
            f.write_str(x)
        } else {
            self.0.fmt(f)
        }
    }
}

#[derive(Debug)]
enum Etc1SChannelType {
    Rgb = 0,
    Rrr = 3,
    Ggg = 4,
    Aaa = 15,
}

impl Etc1SChannelType {
    fn parse(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Rgb),
            3 => Some(Self::Rrr),
            4 => Some(Self::Ggg),
            15 => Some(Self::Aaa),
            _ => None,
        }
    }
}

#[derive(Debug)]
enum UastcChannelType {
    Rgb = 0,
    Rgba = 3,
    Rrr = 4,
    Rrrg = 5,
    Rg = 6,
}

impl UastcChannelType {
    fn parse(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Rgb),
            3 => Some(Self::Rgba),
            4 => Some(Self::Rrr),
            5 => Some(Self::Rrrg),
            6 => Some(Self::Rg),
            _ => None,
        }
    }
}

// https://www.khronos.org/registry/DataFormat/specs/1.3/dataformat.1.3.html#_emphasis_role_strong_emphasis_colorprimaries_emphasis_emphasis
#[derive(Debug)]
enum ColorPrimaries {
    Unspecified = 0,
    Bt709,
}

impl ColorPrimaries {
    fn parse(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Unspecified),
            1 => Some(Self::Bt709),
            _ => None,
        }
    }
}
