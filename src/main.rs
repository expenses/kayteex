use std::collections::HashMap;
use std::fmt;

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

        dbg!(&ktx2.header);

        /*for (compressed, decompressed) in ktx2.compressed_images.iter().zip(ktx2.images_decompressed()) {
            //dbg!((compressed.len(), decompressed.unwrap().len()));
        }*/

        dbg!(&ktx2.key_values, &ktx2.custom_key_values);
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

        let block = DescriptorBlock {
            color_model: ColorModel(this.read_u8()?),
            transfer_function: TransferFunction::parse(this.read_u8()?).unwrap(),
            color_primaries: this.read_u8()?,
            flags: DescriptorBlockFlags {
                bits: this.read_u8()?,
            },
            texel_block_dimensions: this.read_array()?,
            bytes_plane: this.read_array()?,
        };

        dbg!(descriptor_block_header, block);

        //dbg!(descriptor_block_header, block);

        while this.read_bytes - dfd_start != dfd_total_size as usize {
            let sample = SampleInformation {
                bit_offset: this.read_u16()?,
                bit_length: this.read_u8()?,
                channel_type_and_qualifiers: this.read_u8()?,
                sample_position: this.read_array()?,
                sample_upper: this.read_u32()?,
                sample_lower: this.read_u32()?,
            };

            dbg!(ChannelType::parse(sample.channel_type_and_qualifiers));

            dbg!(sample);
        }

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
                flags: ImageFlags { bits: this.read_u32()? },
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

            dbg!(&global_data_header, image_desc);
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
                        Some(character) => SwizzleComponent::parse(character)
                            .ok_or_else(|| anyhow::anyhow!("Invalid swizzle component")),
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
    Zero,
    One,
}

impl SwizzleComponent {
    fn parse(character: char) -> Option<Self> {
        match character {
            'r' => Some(Self::Red),
            'g' => Some(Self::Green),
            'b' => Some(Self::Blue),
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
    compressed_images: Vec<Vec<u8>>,
    key_values: KeyValues,
    custom_key_values: HashMap<String, String>,
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
    transfer_function: TransferFunction,
    color_primaries: u8,
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

#[derive(Debug)]
struct SampleInformation {
    bit_offset: u16,
    bit_length: u8,
    channel_type_and_qualifiers: u8,
    sample_position: [u8; 4],
    sample_upper: u32,
    sample_lower: u32,
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
    // https://www.khronos.org/registry/DataFormat/specs/1.3/dataformat.1.3.html#RGBSDAChannels
    struct ChannelType: u8 {
        const RED = 0;
        const GREEN = 1;
        const BLUE = 2;
        const STENCIL = 4;
        const DEPTH = 8;
        const ALPHA = 16;
    }
}

impl ChannelType {
    fn parse(bits: u8) -> Self {
        match bits {
            13 => Self::STENCIL,
            14 => Self::DEPTH,
            15 => Self::ALPHA,
            _ => Self { bits },
        }
    }
}

/// This is copied from ash to avoid a dependency.
#[derive(PartialEq, Eq)]
pub struct VkFormat(pub u32);

impl VkFormat {
    pub const UNDEFINED: Self = Self(0);
    pub const R4G4_UNORM_PACK8: Self = Self(1);
    pub const R4G4B4A4_UNORM_PACK16: Self = Self(2);
    pub const B4G4R4A4_UNORM_PACK16: Self = Self(3);
    pub const R5G6B5_UNORM_PACK16: Self = Self(4);
    pub const B5G6R5_UNORM_PACK16: Self = Self(5);
    pub const R5G5B5A1_UNORM_PACK16: Self = Self(6);
    pub const B5G5R5A1_UNORM_PACK16: Self = Self(7);
    pub const A1R5G5B5_UNORM_PACK16: Self = Self(8);
    pub const R8_UNORM: Self = Self(9);
    pub const R8_SNORM: Self = Self(10);
    pub const R8_USCALED: Self = Self(11);
    pub const R8_SSCALED: Self = Self(12);
    pub const R8_UINT: Self = Self(13);
    pub const R8_SINT: Self = Self(14);
    pub const R8_SRGB: Self = Self(15);
    pub const R8G8_UNORM: Self = Self(16);
    pub const R8G8_SNORM: Self = Self(17);
    pub const R8G8_USCALED: Self = Self(18);
    pub const R8G8_SSCALED: Self = Self(19);
    pub const R8G8_UINT: Self = Self(20);
    pub const R8G8_SINT: Self = Self(21);
    pub const R8G8_SRGB: Self = Self(22);
    pub const R8G8B8_UNORM: Self = Self(23);
    pub const R8G8B8_SNORM: Self = Self(24);
    pub const R8G8B8_USCALED: Self = Self(25);
    pub const R8G8B8_SSCALED: Self = Self(26);
    pub const R8G8B8_UINT: Self = Self(27);
    pub const R8G8B8_SINT: Self = Self(28);
    pub const R8G8B8_SRGB: Self = Self(29);
    pub const B8G8R8_UNORM: Self = Self(30);
    pub const B8G8R8_SNORM: Self = Self(31);
    pub const B8G8R8_USCALED: Self = Self(32);
    pub const B8G8R8_SSCALED: Self = Self(33);
    pub const B8G8R8_UINT: Self = Self(34);
    pub const B8G8R8_SINT: Self = Self(35);
    pub const B8G8R8_SRGB: Self = Self(36);
    pub const R8G8B8A8_UNORM: Self = Self(37);
    pub const R8G8B8A8_SNORM: Self = Self(38);
    pub const R8G8B8A8_USCALED: Self = Self(39);
    pub const R8G8B8A8_SSCALED: Self = Self(40);
    pub const R8G8B8A8_UINT: Self = Self(41);
    pub const R8G8B8A8_SINT: Self = Self(42);
    pub const R8G8B8A8_SRGB: Self = Self(43);
    pub const B8G8R8A8_UNORM: Self = Self(44);
    pub const B8G8R8A8_SNORM: Self = Self(45);
    pub const B8G8R8A8_USCALED: Self = Self(46);
    pub const B8G8R8A8_SSCALED: Self = Self(47);
    pub const B8G8R8A8_UINT: Self = Self(48);
    pub const B8G8R8A8_SINT: Self = Self(49);
    pub const B8G8R8A8_SRGB: Self = Self(50);
    pub const A8B8G8R8_UNORM_PACK32: Self = Self(51);
    pub const A8B8G8R8_SNORM_PACK32: Self = Self(52);
    pub const A8B8G8R8_USCALED_PACK32: Self = Self(53);
    pub const A8B8G8R8_SSCALED_PACK32: Self = Self(54);
    pub const A8B8G8R8_UINT_PACK32: Self = Self(55);
    pub const A8B8G8R8_SINT_PACK32: Self = Self(56);
    pub const A8B8G8R8_SRGB_PACK32: Self = Self(57);
    pub const A2R10G10B10_UNORM_PACK32: Self = Self(58);
    pub const A2R10G10B10_SNORM_PACK32: Self = Self(59);
    pub const A2R10G10B10_USCALED_PACK32: Self = Self(60);
    pub const A2R10G10B10_SSCALED_PACK32: Self = Self(61);
    pub const A2R10G10B10_UINT_PACK32: Self = Self(62);
    pub const A2R10G10B10_SINT_PACK32: Self = Self(63);
    pub const A2B10G10R10_UNORM_PACK32: Self = Self(64);
    pub const A2B10G10R10_SNORM_PACK32: Self = Self(65);
    pub const A2B10G10R10_USCALED_PACK32: Self = Self(66);
    pub const A2B10G10R10_SSCALED_PACK32: Self = Self(67);
    pub const A2B10G10R10_UINT_PACK32: Self = Self(68);
    pub const A2B10G10R10_SINT_PACK32: Self = Self(69);
    pub const R16_UNORM: Self = Self(70);
    pub const R16_SNORM: Self = Self(71);
    pub const R16_USCALED: Self = Self(72);
    pub const R16_SSCALED: Self = Self(73);
    pub const R16_UINT: Self = Self(74);
    pub const R16_SINT: Self = Self(75);
    pub const R16_SFLOAT: Self = Self(76);
    pub const R16G16_UNORM: Self = Self(77);
    pub const R16G16_SNORM: Self = Self(78);
    pub const R16G16_USCALED: Self = Self(79);
    pub const R16G16_SSCALED: Self = Self(80);
    pub const R16G16_UINT: Self = Self(81);
    pub const R16G16_SINT: Self = Self(82);
    pub const R16G16_SFLOAT: Self = Self(83);
    pub const R16G16B16_UNORM: Self = Self(84);
    pub const R16G16B16_SNORM: Self = Self(85);
    pub const R16G16B16_USCALED: Self = Self(86);
    pub const R16G16B16_SSCALED: Self = Self(87);
    pub const R16G16B16_UINT: Self = Self(88);
    pub const R16G16B16_SINT: Self = Self(89);
    pub const R16G16B16_SFLOAT: Self = Self(90);
    pub const R16G16B16A16_UNORM: Self = Self(91);
    pub const R16G16B16A16_SNORM: Self = Self(92);
    pub const R16G16B16A16_USCALED: Self = Self(93);
    pub const R16G16B16A16_SSCALED: Self = Self(94);
    pub const R16G16B16A16_UINT: Self = Self(95);
    pub const R16G16B16A16_SINT: Self = Self(96);
    pub const R16G16B16A16_SFLOAT: Self = Self(97);
    pub const R32_UINT: Self = Self(98);
    pub const R32_SINT: Self = Self(99);
    pub const R32_SFLOAT: Self = Self(100);
    pub const R32G32_UINT: Self = Self(101);
    pub const R32G32_SINT: Self = Self(102);
    pub const R32G32_SFLOAT: Self = Self(103);
    pub const R32G32B32_UINT: Self = Self(104);
    pub const R32G32B32_SINT: Self = Self(105);
    pub const R32G32B32_SFLOAT: Self = Self(106);
    pub const R32G32B32A32_UINT: Self = Self(107);
    pub const R32G32B32A32_SINT: Self = Self(108);
    pub const R32G32B32A32_SFLOAT: Self = Self(109);
    pub const R64_UINT: Self = Self(110);
    pub const R64_SINT: Self = Self(111);
    pub const R64_SFLOAT: Self = Self(112);
    pub const R64G64_UINT: Self = Self(113);
    pub const R64G64_SINT: Self = Self(114);
    pub const R64G64_SFLOAT: Self = Self(115);
    pub const R64G64B64_UINT: Self = Self(116);
    pub const R64G64B64_SINT: Self = Self(117);
    pub const R64G64B64_SFLOAT: Self = Self(118);
    pub const R64G64B64A64_UINT: Self = Self(119);
    pub const R64G64B64A64_SINT: Self = Self(120);
    pub const R64G64B64A64_SFLOAT: Self = Self(121);
    pub const B10G11R11_UFLOAT_PACK32: Self = Self(122);
    pub const E5B9G9R9_UFLOAT_PACK32: Self = Self(123);
    pub const D16_UNORM: Self = Self(124);
    pub const X8_D24_UNORM_PACK32: Self = Self(125);
    pub const D32_SFLOAT: Self = Self(126);
    pub const S8_UINT: Self = Self(127);
    pub const D16_UNORM_S8_UINT: Self = Self(128);
    pub const D24_UNORM_S8_UINT: Self = Self(129);
    pub const D32_SFLOAT_S8_UINT: Self = Self(130);
    pub const BC1_RGB_UNORM_BLOCK: Self = Self(131);
    pub const BC1_RGB_SRGB_BLOCK: Self = Self(132);
    pub const BC1_RGBA_UNORM_BLOCK: Self = Self(133);
    pub const BC1_RGBA_SRGB_BLOCK: Self = Self(134);
    pub const BC2_UNORM_BLOCK: Self = Self(135);
    pub const BC2_SRGB_BLOCK: Self = Self(136);
    pub const BC3_UNORM_BLOCK: Self = Self(137);
    pub const BC3_SRGB_BLOCK: Self = Self(138);
    pub const BC4_UNORM_BLOCK: Self = Self(139);
    pub const BC4_SNORM_BLOCK: Self = Self(140);
    pub const BC5_UNORM_BLOCK: Self = Self(141);
    pub const BC5_SNORM_BLOCK: Self = Self(142);
    pub const BC6H_UFLOAT_BLOCK: Self = Self(143);
    pub const BC6H_SFLOAT_BLOCK: Self = Self(144);
    pub const BC7_UNORM_BLOCK: Self = Self(145);
    pub const BC7_SRGB_BLOCK: Self = Self(146);
    pub const ETC2_R8G8B8_UNORM_BLOCK: Self = Self(147);
    pub const ETC2_R8G8B8_SRGB_BLOCK: Self = Self(148);
    pub const ETC2_R8G8B8A1_UNORM_BLOCK: Self = Self(149);
    pub const ETC2_R8G8B8A1_SRGB_BLOCK: Self = Self(150);
    pub const ETC2_R8G8B8A8_UNORM_BLOCK: Self = Self(151);
    pub const ETC2_R8G8B8A8_SRGB_BLOCK: Self = Self(152);
    pub const EAC_R11_UNORM_BLOCK: Self = Self(153);
    pub const EAC_R11_SNORM_BLOCK: Self = Self(154);
    pub const EAC_R11G11_UNORM_BLOCK: Self = Self(155);
    pub const EAC_R11G11_SNORM_BLOCK: Self = Self(156);
    pub const ASTC_4X4_UNORM_BLOCK: Self = Self(157);
    pub const ASTC_4X4_SRGB_BLOCK: Self = Self(158);
    pub const ASTC_5X4_UNORM_BLOCK: Self = Self(159);
    pub const ASTC_5X4_SRGB_BLOCK: Self = Self(160);
    pub const ASTC_5X5_UNORM_BLOCK: Self = Self(161);
    pub const ASTC_5X5_SRGB_BLOCK: Self = Self(162);
    pub const ASTC_6X5_UNORM_BLOCK: Self = Self(163);
    pub const ASTC_6X5_SRGB_BLOCK: Self = Self(164);
    pub const ASTC_6X6_UNORM_BLOCK: Self = Self(165);
    pub const ASTC_6X6_SRGB_BLOCK: Self = Self(166);
    pub const ASTC_8X5_UNORM_BLOCK: Self = Self(167);
    pub const ASTC_8X5_SRGB_BLOCK: Self = Self(168);
    pub const ASTC_8X6_UNORM_BLOCK: Self = Self(169);
    pub const ASTC_8X6_SRGB_BLOCK: Self = Self(170);
    pub const ASTC_8X8_UNORM_BLOCK: Self = Self(171);
    pub const ASTC_8X8_SRGB_BLOCK: Self = Self(172);
    pub const ASTC_10X5_UNORM_BLOCK: Self = Self(173);
    pub const ASTC_10X5_SRGB_BLOCK: Self = Self(174);
    pub const ASTC_10X6_UNORM_BLOCK: Self = Self(175);
    pub const ASTC_10X6_SRGB_BLOCK: Self = Self(176);
    pub const ASTC_10X8_UNORM_BLOCK: Self = Self(177);
    pub const ASTC_10X8_SRGB_BLOCK: Self = Self(178);
    pub const ASTC_10X10_UNORM_BLOCK: Self = Self(179);
    pub const ASTC_10X10_SRGB_BLOCK: Self = Self(180);
    pub const ASTC_12X10_UNORM_BLOCK: Self = Self(181);
    pub const ASTC_12X10_SRGB_BLOCK: Self = Self(182);
    pub const ASTC_12X12_UNORM_BLOCK: Self = Self(183);
    pub const ASTC_12X12_SRGB_BLOCK: Self = Self(184);

    pub const PVRTC1_2BPP_UNORM_BLOCK_IMG: Self = Self(1_000_054_000);
    pub const PVRTC1_4BPP_UNORM_BLOCK_IMG: Self = Self(1_000_054_001);
    pub const PVRTC2_2BPP_UNORM_BLOCK_IMG: Self = Self(1_000_054_002);
    pub const PVRTC2_4BPP_UNORM_BLOCK_IMG: Self = Self(1_000_054_003);
    pub const PVRTC1_2BPP_SRGB_BLOCK_IMG: Self = Self(1_000_054_004);
    pub const PVRTC1_4BPP_SRGB_BLOCK_IMG: Self = Self(1_000_054_005);
    pub const PVRTC2_2BPP_SRGB_BLOCK_IMG: Self = Self(1_000_054_006);
    pub const PVRTC2_4BPP_SRGB_BLOCK_IMG: Self = Self(1_000_054_007);

    pub const ASTC_4X4_SFLOAT_BLOCK_EXT: Self = Self(1_000_066_000);
    pub const ASTC_5X4_SFLOAT_BLOCK_EXT: Self = Self(1_000_066_001);
    pub const ASTC_5X5_SFLOAT_BLOCK_EXT: Self = Self(1_000_066_002);
    pub const ASTC_6X5_SFLOAT_BLOCK_EXT: Self = Self(1_000_066_003);
    pub const ASTC_6X6_SFLOAT_BLOCK_EXT: Self = Self(1_000_066_004);
    pub const ASTC_8X5_SFLOAT_BLOCK_EXT: Self = Self(1_000_066_005);
    pub const ASTC_8X6_SFLOAT_BLOCK_EXT: Self = Self(1_000_066_006);
    pub const ASTC_8X8_SFLOAT_BLOCK_EXT: Self = Self(1_000_066_007);
    pub const ASTC_10X5_SFLOAT_BLOCK_EXT: Self = Self(1_000_066_008);
    pub const ASTC_10X6_SFLOAT_BLOCK_EXT: Self = Self(1_000_066_009);
    pub const ASTC_10X8_SFLOAT_BLOCK_EXT: Self = Self(1_000_066_010);
    pub const ASTC_10X10_SFLOAT_BLOCK_EXT: Self = Self(1_000_066_011);
    pub const ASTC_12X10_SFLOAT_BLOCK_EXT: Self = Self(1_000_066_012);
    pub const ASTC_12X12_SFLOAT_BLOCK_EXT: Self = Self(1_000_066_013);

    pub const G8B8G8R8_422_UNORM: Self = Self(1_000_156_000);
    pub const B8G8R8G8_422_UNORM: Self = Self(1_000_156_001);
    pub const G8_B8_R8_3PLANE_420_UNORM: Self = Self(1_000_156_002);
    pub const G8_B8R8_2PLANE_420_UNORM: Self = Self(1_000_156_003);
    pub const G8_B8_R8_3PLANE_422_UNORM: Self = Self(1_000_156_004);
    pub const G8_B8R8_2PLANE_422_UNORM: Self = Self(1_000_156_005);
    pub const G8_B8_R8_3PLANE_444_UNORM: Self = Self(1_000_156_006);
    pub const R10X6_UNORM_PACK16: Self = Self(1_000_156_007);
    pub const R10X6G10X6_UNORM_2PACK16: Self = Self(1_000_156_008);
    pub const R10X6G10X6B10X6A10X6_UNORM_4PACK16: Self = Self(1_000_156_009);
    pub const G10X6B10X6G10X6R10X6_422_UNORM_4PACK16: Self = Self(1_000_156_010);
    pub const B10X6G10X6R10X6G10X6_422_UNORM_4PACK16: Self = Self(1_000_156_011);
    pub const G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16: Self = Self(1_000_156_012);
    pub const G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16: Self = Self(1_000_156_013);
    pub const G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16: Self = Self(1_000_156_014);
    pub const G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16: Self = Self(1_000_156_015);
    pub const G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16: Self = Self(1_000_156_016);
    pub const R12X4_UNORM_PACK16: Self = Self(1_000_156_017);
    pub const R12X4G12X4_UNORM_2PACK16: Self = Self(1_000_156_018);
    pub const R12X4G12X4B12X4A12X4_UNORM_4PACK16: Self = Self(1_000_156_019);
    pub const G12X4B12X4G12X4R12X4_422_UNORM_4PACK16: Self = Self(1_000_156_020);
    pub const B12X4G12X4R12X4G12X4_422_UNORM_4PACK16: Self = Self(1_000_156_021);
    pub const G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16: Self = Self(1_000_156_022);
    pub const G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16: Self = Self(1_000_156_023);
    pub const G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16: Self = Self(1_000_156_024);
    pub const G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16: Self = Self(1_000_156_025);
    pub const G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16: Self = Self(1_000_156_026);
    pub const G16B16G16R16_422_UNORM: Self = Self(1_000_156_027);
    pub const B16G16R16G16_422_UNORM: Self = Self(1_000_156_028);
    pub const G16_B16_R16_3PLANE_420_UNORM: Self = Self(1_000_156_029);
    pub const G16_B16R16_2PLANE_420_UNORM: Self = Self(1_000_156_030);
    pub const G16_B16_R16_3PLANE_422_UNORM: Self = Self(1_000_156_031);
    pub const G16_B16R16_2PLANE_422_UNORM: Self = Self(1_000_156_032);
    pub const G16_B16_R16_3PLANE_444_UNORM: Self = Self(1_000_156_033);

    pub const ASTC_3X3X3_UNORM_BLOCK_EXT: Self = Self(1_000_288_000);
    pub const ASTC_3X3X3_SRGB_BLOCK_EXT: Self = Self(1_000_288_001);
    pub const ASTC_3X3X3_SFLOAT_BLOCK_EXT: Self = Self(1_000_288_002);
    pub const ASTC_4X3X3_UNORM_BLOCK_EXT: Self = Self(1_000_288_003);
    pub const ASTC_4X3X3_SRGB_BLOCK_EXT: Self = Self(1_000_288_004);
    pub const ASTC_4X3X3_SFLOAT_BLOCK_EXT: Self = Self(1_000_288_005);
    pub const ASTC_4X4X3_UNORM_BLOCK_EXT: Self = Self(1_000_288_006);
    pub const ASTC_4X4X3_SRGB_BLOCK_EXT: Self = Self(1_000_288_007);
    pub const ASTC_4X4X3_SFLOAT_BLOCK_EXT: Self = Self(1_000_288_008);
    pub const ASTC_4X4X4_UNORM_BLOCK_EXT: Self = Self(1_000_288_009);
    pub const ASTC_4X4X4_SRGB_BLOCK_EXT: Self = Self(1_000_288_010);
    pub const ASTC_4X4X4_SFLOAT_BLOCK_EXT: Self = Self(1_000_288_011);
    pub const ASTC_5X4X4_UNORM_BLOCK_EXT: Self = Self(1_000_288_012);
    pub const ASTC_5X4X4_SRGB_BLOCK_EXT: Self = Self(1_000_288_013);
    pub const ASTC_5X4X4_SFLOAT_BLOCK_EXT: Self = Self(1_000_288_014);
    pub const ASTC_5X5X4_UNORM_BLOCK_EXT: Self = Self(1_000_288_015);
    pub const ASTC_5X5X4_SRGB_BLOCK_EXT: Self = Self(1_000_288_016);
    pub const ASTC_5X5X4_SFLOAT_BLOCK_EXT: Self = Self(1_000_288_017);
    pub const ASTC_5X5X5_UNORM_BLOCK_EXT: Self = Self(1_000_288_018);
    pub const ASTC_5X5X5_SRGB_BLOCK_EXT: Self = Self(1_000_288_019);
    pub const ASTC_5X5X5_SFLOAT_BLOCK_EXT: Self = Self(1_000_288_020);
    pub const ASTC_6X5X5_UNORM_BLOCK_EXT: Self = Self(1_000_288_021);
    pub const ASTC_6X5X5_SRGB_BLOCK_EXT: Self = Self(1_000_288_022);
    pub const ASTC_6X5X5_SFLOAT_BLOCK_EXT: Self = Self(1_000_288_023);
    pub const ASTC_6X6X5_UNORM_BLOCK_EXT: Self = Self(1_000_288_024);
    pub const ASTC_6X6X5_SRGB_BLOCK_EXT: Self = Self(1_000_288_025);
    pub const ASTC_6X6X5_SFLOAT_BLOCK_EXT: Self = Self(1_000_288_026);
    pub const ASTC_6X6X6_UNORM_BLOCK_EXT: Self = Self(1_000_288_027);
    pub const ASTC_6X6X6_SRGB_BLOCK_EXT: Self = Self(1_000_288_028);
    pub const ASTC_6X6X6_SFLOAT_BLOCK_EXT: Self = Self(1_000_288_029);

    pub const A4R4G4B4_UNORM_PACK16_EXT: Self = Self(1_000_340_000);
    pub const A4B4G4R4_UNORM_PACK16_EXT: Self = Self(1_000_340_001);
}

impl fmt::Debug for VkFormat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let name = match *self {
            Self::UNDEFINED => Some("UNDEFINED"),
            Self::R4G4_UNORM_PACK8 => Some("R4G4_UNORM_PACK8"),
            Self::R4G4B4A4_UNORM_PACK16 => Some("R4G4B4A4_UNORM_PACK16"),
            Self::B4G4R4A4_UNORM_PACK16 => Some("B4G4R4A4_UNORM_PACK16"),
            Self::R5G6B5_UNORM_PACK16 => Some("R5G6B5_UNORM_PACK16"),
            Self::B5G6R5_UNORM_PACK16 => Some("B5G6R5_UNORM_PACK16"),
            Self::R5G5B5A1_UNORM_PACK16 => Some("R5G5B5A1_UNORM_PACK16"),
            Self::B5G5R5A1_UNORM_PACK16 => Some("B5G5R5A1_UNORM_PACK16"),
            Self::A1R5G5B5_UNORM_PACK16 => Some("A1R5G5B5_UNORM_PACK16"),
            Self::R8_UNORM => Some("R8_UNORM"),
            Self::R8_SNORM => Some("R8_SNORM"),
            Self::R8_USCALED => Some("R8_USCALED"),
            Self::R8_SSCALED => Some("R8_SSCALED"),
            Self::R8_UINT => Some("R8_UINT"),
            Self::R8_SINT => Some("R8_SINT"),
            Self::R8_SRGB => Some("R8_SRGB"),
            Self::R8G8_UNORM => Some("R8G8_UNORM"),
            Self::R8G8_SNORM => Some("R8G8_SNORM"),
            Self::R8G8_USCALED => Some("R8G8_USCALED"),
            Self::R8G8_SSCALED => Some("R8G8_SSCALED"),
            Self::R8G8_UINT => Some("R8G8_UINT"),
            Self::R8G8_SINT => Some("R8G8_SINT"),
            Self::R8G8_SRGB => Some("R8G8_SRGB"),
            Self::R8G8B8_UNORM => Some("R8G8B8_UNORM"),
            Self::R8G8B8_SNORM => Some("R8G8B8_SNORM"),
            Self::R8G8B8_USCALED => Some("R8G8B8_USCALED"),
            Self::R8G8B8_SSCALED => Some("R8G8B8_SSCALED"),
            Self::R8G8B8_UINT => Some("R8G8B8_UINT"),
            Self::R8G8B8_SINT => Some("R8G8B8_SINT"),
            Self::R8G8B8_SRGB => Some("R8G8B8_SRGB"),
            Self::B8G8R8_UNORM => Some("B8G8R8_UNORM"),
            Self::B8G8R8_SNORM => Some("B8G8R8_SNORM"),
            Self::B8G8R8_USCALED => Some("B8G8R8_USCALED"),
            Self::B8G8R8_SSCALED => Some("B8G8R8_SSCALED"),
            Self::B8G8R8_UINT => Some("B8G8R8_UINT"),
            Self::B8G8R8_SINT => Some("B8G8R8_SINT"),
            Self::B8G8R8_SRGB => Some("B8G8R8_SRGB"),
            Self::R8G8B8A8_UNORM => Some("R8G8B8A8_UNORM"),
            Self::R8G8B8A8_SNORM => Some("R8G8B8A8_SNORM"),
            Self::R8G8B8A8_USCALED => Some("R8G8B8A8_USCALED"),
            Self::R8G8B8A8_SSCALED => Some("R8G8B8A8_SSCALED"),
            Self::R8G8B8A8_UINT => Some("R8G8B8A8_UINT"),
            Self::R8G8B8A8_SINT => Some("R8G8B8A8_SINT"),
            Self::R8G8B8A8_SRGB => Some("R8G8B8A8_SRGB"),
            Self::B8G8R8A8_UNORM => Some("B8G8R8A8_UNORM"),
            Self::B8G8R8A8_SNORM => Some("B8G8R8A8_SNORM"),
            Self::B8G8R8A8_USCALED => Some("B8G8R8A8_USCALED"),
            Self::B8G8R8A8_SSCALED => Some("B8G8R8A8_SSCALED"),
            Self::B8G8R8A8_UINT => Some("B8G8R8A8_UINT"),
            Self::B8G8R8A8_SINT => Some("B8G8R8A8_SINT"),
            Self::B8G8R8A8_SRGB => Some("B8G8R8A8_SRGB"),
            Self::A8B8G8R8_UNORM_PACK32 => Some("A8B8G8R8_UNORM_PACK32"),
            Self::A8B8G8R8_SNORM_PACK32 => Some("A8B8G8R8_SNORM_PACK32"),
            Self::A8B8G8R8_USCALED_PACK32 => Some("A8B8G8R8_USCALED_PACK32"),
            Self::A8B8G8R8_SSCALED_PACK32 => Some("A8B8G8R8_SSCALED_PACK32"),
            Self::A8B8G8R8_UINT_PACK32 => Some("A8B8G8R8_UINT_PACK32"),
            Self::A8B8G8R8_SINT_PACK32 => Some("A8B8G8R8_SINT_PACK32"),
            Self::A8B8G8R8_SRGB_PACK32 => Some("A8B8G8R8_SRGB_PACK32"),
            Self::A2R10G10B10_UNORM_PACK32 => Some("A2R10G10B10_UNORM_PACK32"),
            Self::A2R10G10B10_SNORM_PACK32 => Some("A2R10G10B10_SNORM_PACK32"),
            Self::A2R10G10B10_USCALED_PACK32 => Some("A2R10G10B10_USCALED_PACK32"),
            Self::A2R10G10B10_SSCALED_PACK32 => Some("A2R10G10B10_SSCALED_PACK32"),
            Self::A2R10G10B10_UINT_PACK32 => Some("A2R10G10B10_UINT_PACK32"),
            Self::A2R10G10B10_SINT_PACK32 => Some("A2R10G10B10_SINT_PACK32"),
            Self::A2B10G10R10_UNORM_PACK32 => Some("A2B10G10R10_UNORM_PACK32"),
            Self::A2B10G10R10_SNORM_PACK32 => Some("A2B10G10R10_SNORM_PACK32"),
            Self::A2B10G10R10_USCALED_PACK32 => Some("A2B10G10R10_USCALED_PACK32"),
            Self::A2B10G10R10_SSCALED_PACK32 => Some("A2B10G10R10_SSCALED_PACK32"),
            Self::A2B10G10R10_UINT_PACK32 => Some("A2B10G10R10_UINT_PACK32"),
            Self::A2B10G10R10_SINT_PACK32 => Some("A2B10G10R10_SINT_PACK32"),
            Self::R16_UNORM => Some("R16_UNORM"),
            Self::R16_SNORM => Some("R16_SNORM"),
            Self::R16_USCALED => Some("R16_USCALED"),
            Self::R16_SSCALED => Some("R16_SSCALED"),
            Self::R16_UINT => Some("R16_UINT"),
            Self::R16_SINT => Some("R16_SINT"),
            Self::R16_SFLOAT => Some("R16_SFLOAT"),
            Self::R16G16_UNORM => Some("R16G16_UNORM"),
            Self::R16G16_SNORM => Some("R16G16_SNORM"),
            Self::R16G16_USCALED => Some("R16G16_USCALED"),
            Self::R16G16_SSCALED => Some("R16G16_SSCALED"),
            Self::R16G16_UINT => Some("R16G16_UINT"),
            Self::R16G16_SINT => Some("R16G16_SINT"),
            Self::R16G16_SFLOAT => Some("R16G16_SFLOAT"),
            Self::R16G16B16_UNORM => Some("R16G16B16_UNORM"),
            Self::R16G16B16_SNORM => Some("R16G16B16_SNORM"),
            Self::R16G16B16_USCALED => Some("R16G16B16_USCALED"),
            Self::R16G16B16_SSCALED => Some("R16G16B16_SSCALED"),
            Self::R16G16B16_UINT => Some("R16G16B16_UINT"),
            Self::R16G16B16_SINT => Some("R16G16B16_SINT"),
            Self::R16G16B16_SFLOAT => Some("R16G16B16_SFLOAT"),
            Self::R16G16B16A16_UNORM => Some("R16G16B16A16_UNORM"),
            Self::R16G16B16A16_SNORM => Some("R16G16B16A16_SNORM"),
            Self::R16G16B16A16_USCALED => Some("R16G16B16A16_USCALED"),
            Self::R16G16B16A16_SSCALED => Some("R16G16B16A16_SSCALED"),
            Self::R16G16B16A16_UINT => Some("R16G16B16A16_UINT"),
            Self::R16G16B16A16_SINT => Some("R16G16B16A16_SINT"),
            Self::R16G16B16A16_SFLOAT => Some("R16G16B16A16_SFLOAT"),
            Self::R32_UINT => Some("R32_UINT"),
            Self::R32_SINT => Some("R32_SINT"),
            Self::R32_SFLOAT => Some("R32_SFLOAT"),
            Self::R32G32_UINT => Some("R32G32_UINT"),
            Self::R32G32_SINT => Some("R32G32_SINT"),
            Self::R32G32_SFLOAT => Some("R32G32_SFLOAT"),
            Self::R32G32B32_UINT => Some("R32G32B32_UINT"),
            Self::R32G32B32_SINT => Some("R32G32B32_SINT"),
            Self::R32G32B32_SFLOAT => Some("R32G32B32_SFLOAT"),
            Self::R32G32B32A32_UINT => Some("R32G32B32A32_UINT"),
            Self::R32G32B32A32_SINT => Some("R32G32B32A32_SINT"),
            Self::R32G32B32A32_SFLOAT => Some("R32G32B32A32_SFLOAT"),
            Self::R64_UINT => Some("R64_UINT"),
            Self::R64_SINT => Some("R64_SINT"),
            Self::R64_SFLOAT => Some("R64_SFLOAT"),
            Self::R64G64_UINT => Some("R64G64_UINT"),
            Self::R64G64_SINT => Some("R64G64_SINT"),
            Self::R64G64_SFLOAT => Some("R64G64_SFLOAT"),
            Self::R64G64B64_UINT => Some("R64G64B64_UINT"),
            Self::R64G64B64_SINT => Some("R64G64B64_SINT"),
            Self::R64G64B64_SFLOAT => Some("R64G64B64_SFLOAT"),
            Self::R64G64B64A64_UINT => Some("R64G64B64A64_UINT"),
            Self::R64G64B64A64_SINT => Some("R64G64B64A64_SINT"),
            Self::R64G64B64A64_SFLOAT => Some("R64G64B64A64_SFLOAT"),
            Self::B10G11R11_UFLOAT_PACK32 => Some("B10G11R11_UFLOAT_PACK32"),
            Self::E5B9G9R9_UFLOAT_PACK32 => Some("E5B9G9R9_UFLOAT_PACK32"),
            Self::D16_UNORM => Some("D16_UNORM"),
            Self::X8_D24_UNORM_PACK32 => Some("X8_D24_UNORM_PACK32"),
            Self::D32_SFLOAT => Some("D32_SFLOAT"),
            Self::S8_UINT => Some("S8_UINT"),
            Self::D16_UNORM_S8_UINT => Some("D16_UNORM_S8_UINT"),
            Self::D24_UNORM_S8_UINT => Some("D24_UNORM_S8_UINT"),
            Self::D32_SFLOAT_S8_UINT => Some("D32_SFLOAT_S8_UINT"),
            Self::BC1_RGB_UNORM_BLOCK => Some("BC1_RGB_UNORM_BLOCK"),
            Self::BC1_RGB_SRGB_BLOCK => Some("BC1_RGB_SRGB_BLOCK"),
            Self::BC1_RGBA_UNORM_BLOCK => Some("BC1_RGBA_UNORM_BLOCK"),
            Self::BC1_RGBA_SRGB_BLOCK => Some("BC1_RGBA_SRGB_BLOCK"),
            Self::BC2_UNORM_BLOCK => Some("BC2_UNORM_BLOCK"),
            Self::BC2_SRGB_BLOCK => Some("BC2_SRGB_BLOCK"),
            Self::BC3_UNORM_BLOCK => Some("BC3_UNORM_BLOCK"),
            Self::BC3_SRGB_BLOCK => Some("BC3_SRGB_BLOCK"),
            Self::BC4_UNORM_BLOCK => Some("BC4_UNORM_BLOCK"),
            Self::BC4_SNORM_BLOCK => Some("BC4_SNORM_BLOCK"),
            Self::BC5_UNORM_BLOCK => Some("BC5_UNORM_BLOCK"),
            Self::BC5_SNORM_BLOCK => Some("BC5_SNORM_BLOCK"),
            Self::BC6H_UFLOAT_BLOCK => Some("BC6H_UFLOAT_BLOCK"),
            Self::BC6H_SFLOAT_BLOCK => Some("BC6H_SFLOAT_BLOCK"),
            Self::BC7_UNORM_BLOCK => Some("BC7_UNORM_BLOCK"),
            Self::BC7_SRGB_BLOCK => Some("BC7_SRGB_BLOCK"),
            Self::ETC2_R8G8B8_UNORM_BLOCK => Some("ETC2_R8G8B8_UNORM_BLOCK"),
            Self::ETC2_R8G8B8_SRGB_BLOCK => Some("ETC2_R8G8B8_SRGB_BLOCK"),
            Self::ETC2_R8G8B8A1_UNORM_BLOCK => Some("ETC2_R8G8B8A1_UNORM_BLOCK"),
            Self::ETC2_R8G8B8A1_SRGB_BLOCK => Some("ETC2_R8G8B8A1_SRGB_BLOCK"),
            Self::ETC2_R8G8B8A8_UNORM_BLOCK => Some("ETC2_R8G8B8A8_UNORM_BLOCK"),
            Self::ETC2_R8G8B8A8_SRGB_BLOCK => Some("ETC2_R8G8B8A8_SRGB_BLOCK"),
            Self::EAC_R11_UNORM_BLOCK => Some("EAC_R11_UNORM_BLOCK"),
            Self::EAC_R11_SNORM_BLOCK => Some("EAC_R11_SNORM_BLOCK"),
            Self::EAC_R11G11_UNORM_BLOCK => Some("EAC_R11G11_UNORM_BLOCK"),
            Self::EAC_R11G11_SNORM_BLOCK => Some("EAC_R11G11_SNORM_BLOCK"),
            Self::ASTC_4X4_UNORM_BLOCK => Some("ASTC_4X4_UNORM_BLOCK"),
            Self::ASTC_4X4_SRGB_BLOCK => Some("ASTC_4X4_SRGB_BLOCK"),
            Self::ASTC_5X4_UNORM_BLOCK => Some("ASTC_5X4_UNORM_BLOCK"),
            Self::ASTC_5X4_SRGB_BLOCK => Some("ASTC_5X4_SRGB_BLOCK"),
            Self::ASTC_5X5_UNORM_BLOCK => Some("ASTC_5X5_UNORM_BLOCK"),
            Self::ASTC_5X5_SRGB_BLOCK => Some("ASTC_5X5_SRGB_BLOCK"),
            Self::ASTC_6X5_UNORM_BLOCK => Some("ASTC_6X5_UNORM_BLOCK"),
            Self::ASTC_6X5_SRGB_BLOCK => Some("ASTC_6X5_SRGB_BLOCK"),
            Self::ASTC_6X6_UNORM_BLOCK => Some("ASTC_6X6_UNORM_BLOCK"),
            Self::ASTC_6X6_SRGB_BLOCK => Some("ASTC_6X6_SRGB_BLOCK"),
            Self::ASTC_8X5_UNORM_BLOCK => Some("ASTC_8X5_UNORM_BLOCK"),
            Self::ASTC_8X5_SRGB_BLOCK => Some("ASTC_8X5_SRGB_BLOCK"),
            Self::ASTC_8X6_UNORM_BLOCK => Some("ASTC_8X6_UNORM_BLOCK"),
            Self::ASTC_8X6_SRGB_BLOCK => Some("ASTC_8X6_SRGB_BLOCK"),
            Self::ASTC_8X8_UNORM_BLOCK => Some("ASTC_8X8_UNORM_BLOCK"),
            Self::ASTC_8X8_SRGB_BLOCK => Some("ASTC_8X8_SRGB_BLOCK"),
            Self::ASTC_10X5_UNORM_BLOCK => Some("ASTC_10X5_UNORM_BLOCK"),
            Self::ASTC_10X5_SRGB_BLOCK => Some("ASTC_10X5_SRGB_BLOCK"),
            Self::ASTC_10X6_UNORM_BLOCK => Some("ASTC_10X6_UNORM_BLOCK"),
            Self::ASTC_10X6_SRGB_BLOCK => Some("ASTC_10X6_SRGB_BLOCK"),
            Self::ASTC_10X8_UNORM_BLOCK => Some("ASTC_10X8_UNORM_BLOCK"),
            Self::ASTC_10X8_SRGB_BLOCK => Some("ASTC_10X8_SRGB_BLOCK"),
            Self::ASTC_10X10_UNORM_BLOCK => Some("ASTC_10X10_UNORM_BLOCK"),
            Self::ASTC_10X10_SRGB_BLOCK => Some("ASTC_10X10_SRGB_BLOCK"),
            Self::ASTC_12X10_UNORM_BLOCK => Some("ASTC_12X10_UNORM_BLOCK"),
            Self::ASTC_12X10_SRGB_BLOCK => Some("ASTC_12X10_SRGB_BLOCK"),
            Self::ASTC_12X12_UNORM_BLOCK => Some("ASTC_12X12_UNORM_BLOCK"),
            Self::ASTC_12X12_SRGB_BLOCK => Some("ASTC_12X12_SRGB_BLOCK"),
            Self::PVRTC1_2BPP_UNORM_BLOCK_IMG => Some("PVRTC1_2BPP_UNORM_BLOCK_IMG"),
            Self::PVRTC1_4BPP_UNORM_BLOCK_IMG => Some("PVRTC1_4BPP_UNORM_BLOCK_IMG"),
            Self::PVRTC2_2BPP_UNORM_BLOCK_IMG => Some("PVRTC2_2BPP_UNORM_BLOCK_IMG"),
            Self::PVRTC2_4BPP_UNORM_BLOCK_IMG => Some("PVRTC2_4BPP_UNORM_BLOCK_IMG"),
            Self::PVRTC1_2BPP_SRGB_BLOCK_IMG => Some("PVRTC1_2BPP_SRGB_BLOCK_IMG"),
            Self::PVRTC1_4BPP_SRGB_BLOCK_IMG => Some("PVRTC1_4BPP_SRGB_BLOCK_IMG"),
            Self::PVRTC2_2BPP_SRGB_BLOCK_IMG => Some("PVRTC2_2BPP_SRGB_BLOCK_IMG"),
            Self::PVRTC2_4BPP_SRGB_BLOCK_IMG => Some("PVRTC2_4BPP_SRGB_BLOCK_IMG"),
            Self::ASTC_4X4_SFLOAT_BLOCK_EXT => Some("ASTC_4X4_SFLOAT_BLOCK_EXT"),
            Self::ASTC_5X4_SFLOAT_BLOCK_EXT => Some("ASTC_5X4_SFLOAT_BLOCK_EXT"),
            Self::ASTC_5X5_SFLOAT_BLOCK_EXT => Some("ASTC_5X5_SFLOAT_BLOCK_EXT"),
            Self::ASTC_6X5_SFLOAT_BLOCK_EXT => Some("ASTC_6X5_SFLOAT_BLOCK_EXT"),
            Self::ASTC_6X6_SFLOAT_BLOCK_EXT => Some("ASTC_6X6_SFLOAT_BLOCK_EXT"),
            Self::ASTC_8X5_SFLOAT_BLOCK_EXT => Some("ASTC_8X5_SFLOAT_BLOCK_EXT"),
            Self::ASTC_8X6_SFLOAT_BLOCK_EXT => Some("ASTC_8X6_SFLOAT_BLOCK_EXT"),
            Self::ASTC_8X8_SFLOAT_BLOCK_EXT => Some("ASTC_8X8_SFLOAT_BLOCK_EXT"),
            Self::ASTC_10X5_SFLOAT_BLOCK_EXT => Some("ASTC_10X5_SFLOAT_BLOCK_EXT"),
            Self::ASTC_10X6_SFLOAT_BLOCK_EXT => Some("ASTC_10X6_SFLOAT_BLOCK_EXT"),
            Self::ASTC_10X8_SFLOAT_BLOCK_EXT => Some("ASTC_10X8_SFLOAT_BLOCK_EXT"),
            Self::ASTC_10X10_SFLOAT_BLOCK_EXT => Some("ASTC_10X10_SFLOAT_BLOCK_EXT"),
            Self::ASTC_12X10_SFLOAT_BLOCK_EXT => Some("ASTC_12X10_SFLOAT_BLOCK_EXT"),
            Self::ASTC_12X12_SFLOAT_BLOCK_EXT => Some("ASTC_12X12_SFLOAT_BLOCK_EXT"),
            Self::ASTC_3X3X3_UNORM_BLOCK_EXT => Some("ASTC_3X3X3_UNORM_BLOCK_EXT"),
            Self::ASTC_3X3X3_SRGB_BLOCK_EXT => Some("ASTC_3X3X3_SRGB_BLOCK_EXT"),
            Self::ASTC_3X3X3_SFLOAT_BLOCK_EXT => Some("ASTC_3X3X3_SFLOAT_BLOCK_EXT"),
            Self::ASTC_4X3X3_UNORM_BLOCK_EXT => Some("ASTC_4X3X3_UNORM_BLOCK_EXT"),
            Self::ASTC_4X3X3_SRGB_BLOCK_EXT => Some("ASTC_4X3X3_SRGB_BLOCK_EXT"),
            Self::ASTC_4X3X3_SFLOAT_BLOCK_EXT => Some("ASTC_4X3X3_SFLOAT_BLOCK_EXT"),
            Self::ASTC_4X4X3_UNORM_BLOCK_EXT => Some("ASTC_4X4X3_UNORM_BLOCK_EXT"),
            Self::ASTC_4X4X3_SRGB_BLOCK_EXT => Some("ASTC_4X4X3_SRGB_BLOCK_EXT"),
            Self::ASTC_4X4X3_SFLOAT_BLOCK_EXT => Some("ASTC_4X4X3_SFLOAT_BLOCK_EXT"),
            Self::ASTC_4X4X4_UNORM_BLOCK_EXT => Some("ASTC_4X4X4_UNORM_BLOCK_EXT"),
            Self::ASTC_4X4X4_SRGB_BLOCK_EXT => Some("ASTC_4X4X4_SRGB_BLOCK_EXT"),
            Self::ASTC_4X4X4_SFLOAT_BLOCK_EXT => Some("ASTC_4X4X4_SFLOAT_BLOCK_EXT"),
            Self::ASTC_5X4X4_UNORM_BLOCK_EXT => Some("ASTC_5X4X4_UNORM_BLOCK_EXT"),
            Self::ASTC_5X4X4_SRGB_BLOCK_EXT => Some("ASTC_5X4X4_SRGB_BLOCK_EXT"),
            Self::ASTC_5X4X4_SFLOAT_BLOCK_EXT => Some("ASTC_5X4X4_SFLOAT_BLOCK_EXT"),
            Self::ASTC_5X5X4_UNORM_BLOCK_EXT => Some("ASTC_5X5X4_UNORM_BLOCK_EXT"),
            Self::ASTC_5X5X4_SRGB_BLOCK_EXT => Some("ASTC_5X5X4_SRGB_BLOCK_EXT"),
            Self::ASTC_5X5X4_SFLOAT_BLOCK_EXT => Some("ASTC_5X5X4_SFLOAT_BLOCK_EXT"),
            Self::ASTC_5X5X5_UNORM_BLOCK_EXT => Some("ASTC_5X5X5_UNORM_BLOCK_EXT"),
            Self::ASTC_5X5X5_SRGB_BLOCK_EXT => Some("ASTC_5X5X5_SRGB_BLOCK_EXT"),
            Self::ASTC_5X5X5_SFLOAT_BLOCK_EXT => Some("ASTC_5X5X5_SFLOAT_BLOCK_EXT"),
            Self::ASTC_6X5X5_UNORM_BLOCK_EXT => Some("ASTC_6X5X5_UNORM_BLOCK_EXT"),
            Self::ASTC_6X5X5_SRGB_BLOCK_EXT => Some("ASTC_6X5X5_SRGB_BLOCK_EXT"),
            Self::ASTC_6X5X5_SFLOAT_BLOCK_EXT => Some("ASTC_6X5X5_SFLOAT_BLOCK_EXT"),
            Self::ASTC_6X6X5_UNORM_BLOCK_EXT => Some("ASTC_6X6X5_UNORM_BLOCK_EXT"),
            Self::ASTC_6X6X5_SRGB_BLOCK_EXT => Some("ASTC_6X6X5_SRGB_BLOCK_EXT"),
            Self::ASTC_6X6X5_SFLOAT_BLOCK_EXT => Some("ASTC_6X6X5_SFLOAT_BLOCK_EXT"),
            Self::ASTC_6X6X6_UNORM_BLOCK_EXT => Some("ASTC_6X6X6_UNORM_BLOCK_EXT"),
            Self::ASTC_6X6X6_SRGB_BLOCK_EXT => Some("ASTC_6X6X6_SRGB_BLOCK_EXT"),
            Self::ASTC_6X6X6_SFLOAT_BLOCK_EXT => Some("ASTC_6X6X6_SFLOAT_BLOCK_EXT"),
            Self::A4R4G4B4_UNORM_PACK16_EXT => Some("A4R4G4B4_UNORM_PACK16_EXT"),
            Self::A4B4G4R4_UNORM_PACK16_EXT => Some("A4B4G4R4_UNORM_PACK16_EXT"),
            Self::G8B8G8R8_422_UNORM => Some("G8B8G8R8_422_UNORM"),
            Self::B8G8R8G8_422_UNORM => Some("B8G8R8G8_422_UNORM"),
            Self::G8_B8_R8_3PLANE_420_UNORM => Some("G8_B8_R8_3PLANE_420_UNORM"),
            Self::G8_B8R8_2PLANE_420_UNORM => Some("G8_B8R8_2PLANE_420_UNORM"),
            Self::G8_B8_R8_3PLANE_422_UNORM => Some("G8_B8_R8_3PLANE_422_UNORM"),
            Self::G8_B8R8_2PLANE_422_UNORM => Some("G8_B8R8_2PLANE_422_UNORM"),
            Self::G8_B8_R8_3PLANE_444_UNORM => Some("G8_B8_R8_3PLANE_444_UNORM"),
            Self::R10X6_UNORM_PACK16 => Some("R10X6_UNORM_PACK16"),
            Self::R10X6G10X6_UNORM_2PACK16 => Some("R10X6G10X6_UNORM_2PACK16"),
            Self::R10X6G10X6B10X6A10X6_UNORM_4PACK16 => Some("R10X6G10X6B10X6A10X6_UNORM_4PACK16"),
            Self::G10X6B10X6G10X6R10X6_422_UNORM_4PACK16 => {
                Some("G10X6B10X6G10X6R10X6_422_UNORM_4PACK16")
            }
            Self::B10X6G10X6R10X6G10X6_422_UNORM_4PACK16 => {
                Some("B10X6G10X6R10X6G10X6_422_UNORM_4PACK16")
            }
            Self::G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16 => {
                Some("G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16")
            }
            Self::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16 => {
                Some("G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16")
            }
            Self::G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16 => {
                Some("G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16")
            }
            Self::G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16 => {
                Some("G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16")
            }
            Self::G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16 => {
                Some("G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16")
            }
            Self::R12X4_UNORM_PACK16 => Some("R12X4_UNORM_PACK16"),
            Self::R12X4G12X4_UNORM_2PACK16 => Some("R12X4G12X4_UNORM_2PACK16"),
            Self::R12X4G12X4B12X4A12X4_UNORM_4PACK16 => Some("R12X4G12X4B12X4A12X4_UNORM_4PACK16"),
            Self::G12X4B12X4G12X4R12X4_422_UNORM_4PACK16 => {
                Some("G12X4B12X4G12X4R12X4_422_UNORM_4PACK16")
            }
            Self::B12X4G12X4R12X4G12X4_422_UNORM_4PACK16 => {
                Some("B12X4G12X4R12X4G12X4_422_UNORM_4PACK16")
            }
            Self::G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16 => {
                Some("G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16")
            }
            Self::G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16 => {
                Some("G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16")
            }
            Self::G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16 => {
                Some("G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16")
            }
            Self::G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16 => {
                Some("G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16")
            }
            Self::G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16 => {
                Some("G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16")
            }
            Self::G16B16G16R16_422_UNORM => Some("G16B16G16R16_422_UNORM"),
            Self::B16G16R16G16_422_UNORM => Some("B16G16R16G16_422_UNORM"),
            Self::G16_B16_R16_3PLANE_420_UNORM => Some("G16_B16_R16_3PLANE_420_UNORM"),
            Self::G16_B16R16_2PLANE_420_UNORM => Some("G16_B16R16_2PLANE_420_UNORM"),
            Self::G16_B16_R16_3PLANE_422_UNORM => Some("G16_B16_R16_3PLANE_422_UNORM"),
            Self::G16_B16R16_2PLANE_422_UNORM => Some("G16_B16R16_2PLANE_422_UNORM"),
            Self::G16_B16_R16_3PLANE_444_UNORM => Some("G16_B16_R16_3PLANE_444_UNORM"),
            _ => None,
        };
        if let Some(x) = name {
            f.write_str(x)
        } else {
            self.0.fmt(f)
        }
    }
}
